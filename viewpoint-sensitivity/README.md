# Viewpoint Sensitivity Analysis

Motivating 3D reconstruction / perspective-change work by measuring how much
open-vocabulary detection/segmentation quality degrades as viewing
elevation moves away from a canonical "eye-level" angle.

Rendering setup is based on
["Active Open-Vocabulary Recognition"](https://arxiv.org/abs/2311.17938)
(Chen et al., 2023), which renders ShapeNetCore objects in habitat-sim from a
discretized viewing sphere (30-degree azimuth/elevation steps, up to 3.0m
camera distance) to show that CLIP's open-vocabulary accuracy is highly
viewpoint-dependent (29.6% baseline vs. 53.3% with multi-view fusion, no
fine-tuning). This project adapts that same rendering setup to open-vocabulary
*detection/segmentation* models instead of pure classification.

## Pipeline

1. **Controlled multi-view dataset** (this step) — habitat-sim renders each
   object, alone on a bare stage, from a fixed grid of elevations x azimuths;
   a naive 4-model isolated-server setup + orchestrator plumbing is stood up
   alongside it. Object source is either ShapeNetCore or Objaverse (LVIS
   subset) — see `rendering/config.py:DATASET_SOURCE`. ShapeNet's HF dataset
   is gated (license approval can take a while); Objaverse needs no login, so
   it's the current default until ShapeNet access comes through. Both produce
   the same manifest schema, so switching later is a one-line config change
   plus re-running the render.
2. Fixed text prompt per object category + fixed decision threshold across
   all models/views (see `rendering/config.py:CATEGORY_PROMPTS` and
   `model_servers/common/schema.py:DetectRequest.threshold`).
3. Per (object instance x elevation x model): run inference, log
   score/box/mask/correctness against the manifest's ground-truth box.
4. Naive multi-view fusion baseline vs. any single fixed viewpoint.

## Setup order, end to end

```bash
# 0. install conda if you don't have it, then:
./scripts/setup_envs.sh                 # creates all 6 conda envs

# 1. object dataset -- Objaverse needs no login and is the current default
#    (config.py:DATASET_SOURCE = "objaverse"):
conda activate vps-render
./scripts/download_objaverse.sh --dry-run   # shows matched LVIS keys + counts, downloads nothing
./scripts/download_objaverse.sh              # downloads into ~/.objaverse's own cache

#    Once ShapeNet's gated dataset license is approved, switch
#    config.py:DATASET_SOURCE to "shapenet" and instead:
#      https://huggingface.co/datasets/ShapeNet/shapenetcore-glb  (accept license)
#      export HF_TOKEN=hf_...
#      ./scripts/download_shapenet.sh
#    SAM3's checkpoint is separately gated regardless of which object dataset
#    you use -- accept https://huggingface.co/facebook/sam3 and
#    `huggingface-cli login --token "$HF_TOKEN"` on whatever machine runs its
#    server. GroundingDINO, OWLv2, and YOLO-World need no login at all.

# 2. rendering (step 1 of the sensitivity analysis)
python rendering/render_multiview.py --dry-run   # sanity check model/view counts
python rendering/render_multiview.py             # writes data/renders/**.png + manifest.jsonl

# 3. model servers -- each in its own conda env. The FIRST /detect call to each
#    triggers that model's own (multi-GB) weight download; expect the first
#    request per model to be slow.
./orchestrator/launch_servers.sh start
conda activate vps-orchestrator
python orchestrator/health_check.py     # confirms all 4 servers + manifest are ready
python orchestrator/detect_client.py --model yoloworld --image <path from manifest> --prompt "a chair"

# 4. run inference across every (view x model) -- step 3 of the sensitivity analysis
python orchestrator/run_inference.py    # writes data/renders/inference_log.jsonl (resumable)

# 5. naive multi-view fusion baseline -- step 4
python orchestrator/fusion_baseline.py  # writes data/renders/fusion_summary*.csv

./orchestrator/launch_servers.sh stop
```

## Layout

```
environment/        conda env specs (1 render + 4 model-server + 1 orchestrator)
scripts/             setup_envs.sh, download_shapenet.sh, download_objaverse.sh
rendering/           habitat-sim capture: config.py, habitat_object_loader.py (shared),
                     shapenet_objects.py, objaverse_objects.py, download_objaverse.py, render_multiview.py
model_servers/       one dir per model (yoloworld, groundingdino, owlv2, sam3), FastAPI /health + /detect
orchestrator/        ports.yaml, launch_servers.sh, health_check.py, detect_client.py,
                     run_inference.py (step 3), fusion_baseline.py (step 4), iou.py
data/                shapenet/ or objaverse/ (dataset manifests; ShapeNet .glb files are stored
                     here directly, Objaverse's are cached under ~/.objaverse instead),
                     renders/ (dataset + manifest.jsonl) — all gitignored
```

## What "success" means

Two experiments live in this repo, and they define success in *opposite*
ways. Getting this backwards is an easy mistake to make, so it's spelled out
explicitly here (and in `orchestrator/run_inference.py`'s docstring):

- **True-positive experiments** (the main elevation/azimuth sweep, and the
  `--strategy true` offset grid): the real category is prompted, and
  `correct` means the model's top detection has IoU ≥ 0.5 against the
  manifest's real ground-truth box. Higher is better -- the model correctly
  recognizing what's actually there.
- **False-positive experiment** (the `--strategy false` offset grid, run via
  `run_inference.py --false-positive`): a *wrong*, visually-confusable
  category is prompted instead (`config.py:FALSE_CATEGORY_MAP`). There is no
  ground-truth box for a category that isn't in the image, so there is no
  "correct" -- instead `false_positive` means the model returned *any*
  detection at all for that wrong prompt. Here, lower is better: a detection
  means the model got fooled. This is why the field is never called
  `correct` in false-positive mode -- reusing that name with an inverted
  meaning would be a silent footgun for anyone reading the logs later.

## Offset-grid experiments (does frame POSITION, not just viewing angle, matter?)

Distinct from the main elevation/azimuth sweep (which always keeps the
object dead-centered), `rendering/render_framing_grid.py` tests whether an
object's *position within the frame* affects detection quality. Starting
from a selected seed viewpoint, the camera is translated (never rotated) in
a 12x12 XY grid so the object drifts from center toward each corner via pure
parallax -- see `rendering/camera_offset.py`. Two independent concerns, kept
in separate modules on purpose:

- **`rendering/viewpoint_selection.py`** decides *which* seed viewpoint to
  start from. `select_best_true_viewpoints()` picks each instance's most
  confidently *correctly*-recognized viewpoint (one seed per instance, shared
  across all 4 models). `select_false_positive_seed_viewpoints(log, model)`
  picks, for ONE model, its own genuinely-confirmed false positives (real
  `false_positive=True` rows for that model, not an averaged confidence
  across models -- different models fail on different instances at
  different viewpoints) -- "after this model already found the alleged
  [wrong] object here, does moving toward a corner make it worse or does it
  self-correct." Run once per model; an instance that model never
  false-positived on simply isn't in its seed list.
- **`rendering/render_framing_grid.py` + `rendering/camera_offset.py`** don't
  care which strategy picked the seed -- they just sweep the XY grid around
  whatever viewpoint they're given.

```bash
# true-positive offset grid (default) -- one seed per instance, shared across models
conda activate vps-render
python rendering/render_framing_grid.py                 # all instances
conda activate vps-orchestrator
python orchestrator/run_inference.py \
    --manifest data/renders_framing/framing_manifest.jsonl \
    --output data/renders_framing/framing_inference_log.jsonl
python orchestrator/framing_heatmap.py --per-instance
python orchestrator/framing_image_grid.py --model groundingdino

# false-positive offset grid -- per model. First, a false-prompt inference pass
# over the MAIN dataset finds each model's own real false positives:
conda activate vps-orchestrator
python orchestrator/run_inference.py --false-positive \
    --manifest data/renders/manifest.jsonl \
    --output data/renders/false_positive_inference_log.jsonl

# Then, for EACH model separately (seed sets differ in size -- e.g. GroundingDINO
# fails on ~all 155 instances, OWLv2 on a couple dozen -- so this is 4 separate
# render+inference passes, not one shared one). Sensitivity analysis only: does
# the false positive strengthen/weaken/disappear as the object moves toward a
# corner from the model's own real failure point -- not a true-prompt recovery
# check (that's a separate, not-currently-built question).
for MODEL in groundingdino owlv2 sam3 yoloworld; do
  conda activate vps-render
  python rendering/render_framing_grid.py --strategy false --seed-model $MODEL \
      --false-inference-log data/renders/false_positive_inference_log.jsonl

  conda activate vps-orchestrator
  python orchestrator/run_inference.py --false-positive --models $MODEL \
      --manifest data/renders_framing_falsepositive/$MODEL/framing_manifest.jsonl \
      --output data/renders_framing_falsepositive/$MODEL/framing_inference_log.jsonl

  python orchestrator/framing_heatmap.py --value-field false_positive --per-instance \
      --manifest data/renders_framing_falsepositive/$MODEL/framing_manifest.jsonl \
      --inference-log data/renders_framing_falsepositive/$MODEL/framing_inference_log.jsonl
  python orchestrator/framing_image_grid.py --model $MODEL --value-field false_positive \
      --manifest data/renders_framing_falsepositive/$MODEL/framing_manifest.jsonl \
      --inference-log data/renders_framing_falsepositive/$MODEL/framing_inference_log.jsonl
done
```

## Why isolated envs

YOLO-World (ultralytics), GroundingDINO/OWLv2 (transformers), and SAM3
(bleeding-edge transformers, released Nov 2025, needs a newer version than the
other two) have overlapping but version-incompatible dependency needs. Each
gets its own conda env and talks to the orchestrator only over HTTP
(`/health`, `/detect`), so upgrading one model's stack can never break
another's.

## Known gaps to fill in before running for real

- `scripts/download_shapenet.sh`'s `--include "<synset>/*"` pattern assumes a
  `<synset_id>/` top-level prefix in the `ShapeNet/shapenetcore-glb` HF repo —
  confirm against the actual file listing before a large download.
- `rendering/shapenet_objects.py`'s rigid-object/template API calls (and the
  `cumulative_bb` AABB accessor) are version-sensitive across habitat-sim
  releases — verify against your installed `habitat-sim=0.3.*` if you hit an
  `AttributeError`.
- The 4 model-server envs install torch/torchvision via `pip`, not conda (conda's
  `pytorch` + `nvidia` + `conda-forge` channels reliably `ClobberError` on shared
  C-library paths, e.g. `jpeg` vs. `libjpeg-turbo`). The PyPI wheel bundles its
  own CUDA runtime and auto-detects the GPU; on a CPU-only machine it falls back
  to CPU automatically, no edit needed. Non-NVIDIA `render_habitat.yml` still
  needs its `headless`/GPU note below.
- Non-NVIDIA machines: edit `environment/render_habitat.yml` to use the
  CPU/headless habitat-sim build (drop `headless` if a display *is* attached).
- SAM3's exact `transformers` version pin in `environment/server_sam3.yml`
  should be confirmed against the current release notes at setup time.
