#!/usr/bin/env python
"""Run the large grounding models (LocateAnything, Rex-Omni, Gemini, GPT) on any
render manifest and write logs in the SAME format as orchestrator/run_inference.py,
so confusion_metrics.py scores them next to GroundingDINO / OWLv2 / SAM3 / YOLO-World.

The model runs in-process (no server). Each model gets its own log file next to
the existing ones (suffix _<model>), which confusion_metrics.py picks up:

  main sweep   data/renders/inference_log_<model>.jsonl                     (true prompt)
               data/renders/false_positive_inference_log_<model>.jsonl      (--false-positive)
  xy grids     data/renders_framing_falsepositive/<grid>/framing_true_inference_log_<model>.jsonl
               data/renders_framing_falsepositive/<grid>/framing_inference_log_<model>.jsonl
  benchmark    data/renders_benchmark/benchmark_inference_log_<model>.jsonl
  (pass --output to override; the default is derived from --manifest)

Examples (run from viewpoint-sensitivity/):

    python run_vlm_inference.py --model locate_anything --manifest data/renders/manifest.jsonl --limit 5 --show-raw
    python run_vlm_inference.py --model locate_anything --manifest data/renders/manifest.jsonl
    python run_vlm_inference.py --model locate_anything --manifest data/renders/manifest.jsonl --false-positive
    python run_vlm_inference.py --model gemini --manifest data/renders/manifest.jsonl --instances-per-category 5

Resumable: (view_id, model) pairs already in the output file are skipped.

Scoring notes (keep these in the paper's method section):
  * These models emit boxes without a confidence score, so best_score is 1.0 when a
    box is returned and there is no threshold to tune: they are compared at their
    native operating point. "best_box" is the FIRST box the model emits.
  * Prompt text is the manifest's own prompt (config.CATEGORY_PROMPTS), identical to
    the other models; --false-positive swaps in config.FALSE_CATEGORY_MAP's word.
  * IoU is computed here with continuous xyxy boxes against gt_bbox_xyxy.
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
NO_ENV_FIX = False          # set by --no-env-fix
# stop at the first credential error instead of failing every remaining image
AUTH_ERROR = re.compile(r"API key not valid|API_KEY_INVALID|PERMISSION_DENIED|invalid_api_key|Incorrect API key|"
                        r"401 Unauthorized|Error code: 401|gated repo|You must have access|model_not_found|"
                        r"does not exist or you do not have access|insufficient_quota|credit_balance_exhausted|"
                        r"billing_hard_limit|exceeded your current quota", re.I)
# temporary overload / rate limiting: wait and retry instead of logging a failure
TRANSIENT_ERROR = re.compile(r"\b429\b|rate.?limit|RESOURCE_EXHAUSTED|\b503\b|\b502\b|overloaded|"
                             r"UNAVAILABLE|timed? ?out|Connection reset", re.I)


def detect_with_retry(adapter, pil, prompt, tries=6):
    """Retry transient API errors (rate limits, overload) with exponential backoff."""
    for k in range(tries):
        try:
            return adapter.detect(pil, prompt)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if AUTH_ERROR.search(msg) or not TRANSIENT_ERROR.search(msg) or k == tries - 1:
                raise
            wait = min(120, 5 * 2 ** k)
            print(f"  transient API error ({msg[:90]}...), retrying in {wait}s")
            time.sleep(wait)
AUTH_HELP = {
    "gemini": ("Gemini rejected the API key. Create one at https://aistudio.google.com/apikey, then:\n"
               "  unset GOOGLE_API_KEY                      # google-genai prefers this one if both are set\n"
               "  export GEMINI_API_KEY='AIza...your key...'\n"
               "  python -c \"from google import genai; print(genai.Client().models.generate_content("
               "model='gemini-2.5-flash', contents='say ok').text)\"   # must print ok\n"
               "then re-run the same command (failed rows are retried)."),
    "credits": ("The API account has no credits left (prepaid API billing is separate from a ChatGPT/Gemini\n"
                "subscription). Add credits, e.g. https://platform.openai.com/settings/organization/billing/ ,\n"
                "wait a minute for it to activate, then re-run the same command (failed rows are retried)."),
    "gpt_astra": ("OpenAI rejected the request: invalid OPENAI_API_KEY, or this API account/project has no\n"
                  "access to gpt-6-astra yet (check https://platform.openai.com/settings -> Limits / models).\n"
                  "  export OPENAI_API_KEY='sk-...your key...'\n"
                  "then re-run the same command (failed rows are retried)."),
    "gpt": ("OpenAI rejected the API key. Create one at https://platform.openai.com/api-keys, then:\n"
            "  export OPENAI_API_KEY='sk-...your key...'\n"
            "then re-run the same command (failed rows are retried)."),
}
PASSTHROUGH = ["elevation_deg", "azimuth_deg", "offset_x_deg", "offset_y_deg", "base_elevation_deg",
               "base_azimuth_deg", "offset_el_deg", "offset_az_deg", "true_category", "query_category",
               "is_target", "base_view_id"]



# ---------------------------------------------------------- environment check
def _major_minor(v):
    nums = re.findall(r"\d+", v or "")
    return tuple(int(x) for x in nums[:2]) if len(nums) >= 2 else None


def ensure_transformers_compat(model_name, allow_fix=True):
    """LocateAnything ships remote code written for an older transformers release; newer
    majors fail at load (allow_all_kernels, rope_theta, ...). Compare the installed version
    with the one recorded in the model's config.json and, if the major version differs,
    install the recorded version into the ACTIVE conda env and restart this script."""
    import os
    import subprocess
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        installed = None
    saved = None
    try:
        from huggingface_hub import hf_hub_download
        saved = json.load(open(hf_hub_download(model_name, "config.json"))).get("transformers_version")
    except Exception as e:
        print(f"  (could not read {model_name}/config.json: {e})")
    inst_v, saved_v = _major_minor(installed), _major_minor(saved)
    print(f"  transformers installed: {installed} | model saved with: {saved or 'unknown'}")
    if inst_v is None:
        target = f"transformers=={saved}" if saved else "transformers<5"
    elif saved_v is not None and inst_v[0] != saved_v[0]:
        target = f"transformers=={saved}"
    elif saved_v is None and inst_v[0] >= 5:
        target = "transformers<5"
    else:
        return                                         # compatible: nothing to do
    cmd = [sys.executable, "-m", "pip", "install", target]
    if not allow_fix or os.environ.get("VPS_ENV_FIX_DONE"):
        sys.exit(f"transformers {installed} is incompatible with {model_name}. Run:\n  {' '.join(cmd)}")
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not prefix or os.path.basename(prefix) in ("base", "anaconda3", "miniconda3", "miniforge3", "conda"):
        sys.exit("refusing to change packages outside a dedicated conda env: "
                 f"run 'conda activate vps-locate' first, then re-run (or install manually: {' '.join(cmd)})")
    print(f"  -> installing {target} into {prefix} (only this env is changed), then restarting")
    subprocess.check_call(cmd)
    os.environ["VPS_ENV_FIX_DONE"] = "1"               # never loop: one attempt per run
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ------------------------------------------------------------------ adapters
# Each adapter: .name, and .detect(pil_image, prompt) -> (list of xyxy pixel boxes, raw_text)

class LocateAnything:
    """nvidia/LocateAnything-3B. Ported from the lab's own DC-ObjectNav detector
    (DCON/src/perception/obj_detection.py, LocateAnythingDetector, before its removal)."""
    name = "locate_anything"

    def __init__(self, model_name="nvidia/LocateAnything-3B", device="cuda", max_new_tokens=128,
                 repetition_penalty=1.3):
        ensure_transformers_compat(model_name, allow_fix=not NO_ENV_FIX)
        import glob
        import os
        import torch
        # the downloaded remote code uses `X | None` hints, which break on Python 3.9
        for f in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/"
                                              "nvidia/LocateAnything_hyphen_3B/*/*.py")):
            os.system(f"grep -q 'from __future__ import annotations' {f} || "
                      f"sed -i '1s/^/from __future__ import annotations\\n/' {f}")
        from transformers import AutoProcessor, AutoTokenizer
        self.torch, self.device = torch, device
        self.max_new_tokens, self.repetition_penalty = int(max_new_tokens), float(repetition_penalty)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor.tokenizer = self.tokenizer
        self.model = self._load_model(model_name, torch).to(device).eval()

    @staticmethod
    def _load_model(model_name, torch):
        """Load the remote model class, first making its old-style override of
        _check_and_adjust_attn_implementation accept keyword arguments that newer
        transformers releases pass (e.g. allow_all_kernels). Falls back to AutoModel."""
        from transformers import AutoConfig, AutoModel
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], model_name)
            # the repo ships several modules (its own modeling_qwen2.py too) that each override the
            # method with the old signature: patch every class in every downloaded remote module
            n = 0
            for mod_name, mod in list(sys.modules.items()):
                if not mod_name.startswith("transformers_modules") or mod is None:
                    continue
                for klass in list(vars(mod).values()):
                    if not isinstance(klass, type) or klass.__module__ != mod_name:
                        continue
                    orig = klass.__dict__.get("_check_and_adjust_attn_implementation")
                    if orig is None or getattr(orig, "_kw_tolerant", False):
                        continue

                    def patched(self, *args, _orig=orig, **kwargs):
                        try:
                            return _orig(self, *args, **kwargs)
                        except TypeError as e:
                            if "unexpected keyword argument" not in str(e):
                                raise
                            for k in [k for k in kwargs if k != "is_init_check"]:
                                kwargs.pop(k)     # newer keywords the old override doesn't know
                            return _orig(self, *args, **kwargs)

                    patched._kw_tolerant = True
                    klass._check_and_adjust_attn_implementation = patched
                    n += 1
            print(f"  patched {n} remote class(es) for the newer transformers attention signature")
            return cls.from_pretrained(model_name, config=cfg, torch_dtype=torch.bfloat16)
        except (KeyError, AttributeError, ImportError) as e:
            print(f"  (attn-signature patch skipped: {e}; loading with AutoModel)")
            return AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    def detect(self, pil, prompt):
        w, h = pil.size
        text_prompt = f"Locate all the instances that matches the following description: {prompt}."
        messages = [{"role": "user", "content": [{"type": "image", "image": pil},
                                                 {"type": "text", "text": text_prompt}]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            # generate() returns an already-decoded string; it often repeats boxes until
            # max_new_tokens, and the first box is the real one (later ones degenerate)
            response = self.model.generate(
                pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer, max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty, use_cache=True, generation_mode="hybrid")
        boxes = []
        for g in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", response):
            x1, y1, x2, y2 = (int(v) / 1000.0 for v in g)
            xs, ys = sorted((x1 * w, x2 * w)), sorted((y1 * h, y2 * h))
            boxes.append([xs[0], ys[0], xs[1], ys[1]])
        return boxes[:1] if boxes else [], response   # first box only: the tail is repeats


class RexOmni:
    """IDEA-Research/Rex-Omni (Qwen2.5-VL based), via its `rex_omni` package (checked against
    its wrapper.py: inference() returns [{"extracted_predictions": {cat: [{"type": "box",
    "coords": [x0, y0, x1, y1]}]}, "raw_output": ...}], coords in absolute pixels)."""
    name = "rex_omni"

    def __init__(self, model_name="IDEA-Research/Rex-Omni", device="cuda"):
        try:
            from rex_omni import RexOmniWrapper
        except ImportError:
            sys.exit("rex_omni is not installed in this env. Install it (no flash-attn / vllm needed):\n"
                     "  git clone https://github.com/IDEA-Research/Rex-Omni ~/projects/Rex-Omni\n"
                     "  pip install -e ~/projects/Rex-Omni --no-deps\n"
                     "  pip install 'transformers==4.51.3' 'qwen_vl_utils==0.0.14' accelerate 'numpy<2' pillow")
        import importlib.util
        # Rex-Omni defaults to flash_attention_2; fall back to PyTorch SDPA when flash-attn is absent
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        print(f"  Rex-Omni attention: {attn}")
        self.model = RexOmniWrapper(model_path=model_name, backend="transformers", max_tokens=2048,
                                    temperature=0.0, top_p=0.05, top_k=1, repetition_penalty=1.05,
                                    attn_implementation=attn)

    @staticmethod
    def _word(prompt):
        return re.sub(r"^(a|an|the)\s+", "", prompt.strip(), flags=re.I)

    def _parse(self, result, w, h):
        preds = result.get("extracted_predictions", {}) if isinstance(result, dict) else {}
        boxes = []
        for _, items in preds.items():
            for it in items or []:
                c = it.get("coords") if isinstance(it, dict) else None
                if it.get("type", "box") == "box" and c and len(c) == 4:
                    if max(c) <= 1.0:                 # normalized -> pixels
                        c = [c[0] * w, c[1] * h, c[2] * w, c[3] * h]
                    boxes.append([float(min(c[0], c[2])), float(min(c[1], c[3])),
                                  float(max(c[0], c[2])), float(max(c[1], c[3]))])
        return boxes

    def detect(self, pil, prompt):
        w, h = pil.size
        res = self.model.inference(images=pil, task="detection", categories=[self._word(prompt)])
        r0 = res[0] if isinstance(res, (list, tuple)) else res
        raw = r0.get("raw_output", str(r0)) if isinstance(r0, dict) else str(r0)
        return self._parse(r0, w, h), str(raw)


def _vlm_prompt(prompt):
    word = re.sub(r"^(a|an|the)\s+", "", prompt.strip(), flags=re.I)
    return word, (f"Is there {prompt} in this image? If yes, return the bounding box of every {word}. "
                  f"If there is no {word}, return an empty list. Answer with JSON only.")


class Gemini:
    """Google Gemini via the google-genai SDK (pip install google-genai; export GEMINI_API_KEY).
    Gemini returns boxes as [ymin, xmin, ymax, xmax] normalized to 0-1000."""
    name = "gemini"

    def __init__(self, model_name="gemini-2.5-flash", **_):
        from google import genai
        from google.genai import types
        self.types, self.client, self.model_name = types, genai.Client(), model_name

    def detect(self, pil, prompt):
        w, h = pil.size
        word, q = _vlm_prompt(prompt)
        q += ' Format: [{"label": "' + word + '", "box_2d": [ymin, xmin, ymax, xmax]}] normalized to 0-1000.'
        cfg = self.types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        resp = self.client.models.generate_content(model=self.model_name, contents=[pil, q], config=cfg)
        raw = resp.text or ""
        boxes = []
        try:
            items = json.loads(raw)
            items = items if isinstance(items, list) else items.get("boxes", [])
            for it in items:
                b = it.get("box_2d") if isinstance(it, dict) else None
                if b and len(b) == 4:
                    y0, x0, y1, x1 = (float(v) / 1000.0 for v in b)
                    boxes.append([min(x0, x1) * w, min(y0, y1) * h, max(x0, x1) * w, max(y0, y1) * h])
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return boxes, raw


class OpenAIVLM:
    """OpenAI GPT models (pip install openai; export OPENAI_API_KEY). Their boxes are coarse;
    prefer the 'presence' columns in confusion_metrics.py for this model."""
    name = "gpt"

    def __init__(self, model_name="gpt-4o", reasoning_effort=None, **_):
        from openai import OpenAI
        self.client, self.model_name = OpenAI(), model_name
        self.reasoning_effort = reasoning_effort

    def detect(self, pil, prompt):
        w, h = pil.size
        word, q = _vlm_prompt(prompt)
        q += (f' The image is {w}x{h} pixels. Format: {{"boxes": [[x1, y1, x2, y2], ...]}} in pixel '
              f'coordinates, or {{"boxes": []}}.')
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        kw = dict(model=self.model_name, response_format={"type": "json_object"},
                  messages=[{"role": "user", "content": [{"type": "text", "text": q},
                                                         {"type": "image_url", "image_url": {"url": url}}]}])
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self.client.chat.completions.create(temperature=0, **kw)
        except Exception as e:                # some reasoning models reject temperature
            if "temperature" not in str(e):
                raise
            resp = self.client.chat.completions.create(**kw)
        raw = resp.choices[0].message.content or ""
        boxes = []
        try:
            for b in json.loads(raw).get("boxes", []):
                if len(b) == 4:
                    boxes.append([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])])
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return boxes, raw


class GPT6Astra(OpenAIVLM):
    """OpenAI GPT-6 Astra (API id gpt-6-astra, image input; Sept 2026). Same prompt and parsing as
    the GPT adapter. Reasoning effort defaults to 'low' to keep latency/cost down for ~10k images;
    override with --reasoning-effort (low/medium/high/xhigh/max) and report the value used."""
    name = "gpt_astra"

    def __init__(self, model_name="gpt-6-astra", reasoning_effort="low", **kw):
        super().__init__(model_name=model_name, reasoning_effort=reasoning_effort, **kw)


class CLIPClassifier:
    """CLIP zero-shot classifier (no boxes). Each image is classified over the full category list
    with the template "a photo of a <word>."; a query for <word> counts as a detection iff CLIP's
    top-1 label is <word>. So the right-word pass gives TP/FN and the wrong-word pass gives FP/TN,
    exactly like the detectors. It returns a whole-image box only so the logs keep the same shape:
    score CLIP with the PRESENCE columns (results_table.py does this automatically)."""
    name = "clip"

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda", labels=None, **_):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if not labels:
            sys.exit("CLIP needs the category list (taken from the manifest automatically)")
        self.torch, self.device, self.labels = torch, device, list(labels)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        with torch.no_grad():
            t = self.processor(text=[f"a photo of a {w}." for w in self.labels], return_tensors="pt",
                               padding=True).to(device)
            e = self.model.get_text_features(**t)
            self.text = e / e.norm(dim=-1, keepdim=True)
        self._cache = {}
        print(f"  CLIP labels ({len(self.labels)}): {self.labels}")

    def _probs(self, pil):
        key = hash(pil.tobytes())
        if key not in self._cache:
            with self.torch.no_grad():
                px = self.processor(images=pil, return_tensors="pt").to(self.device)
                e = self.model.get_image_features(**px)
                e = e / e.norm(dim=-1, keepdim=True)
                logits = self.model.logit_scale.exp() * e @ self.text.T
                self._cache = {key: logits.softmax(-1)[0].float().cpu().numpy()}   # keep only the last image
        return self._cache[key]

    def detect(self, pil, prompt):
        w, h = pil.size
        word = re.sub(r"^(a|an|the)\s+", "", prompt.strip(), flags=re.I)
        if word not in self.labels:
            raise ValueError(f"'{word}' is not in CLIP's label set {self.labels}")
        p = self._probs(pil)
        top = self.labels[int(p.argmax())]
        raw = json.dumps({"top1": top, "p_top1": round(float(p.max()), 4),
                          "p_query": round(float(p[self.labels.index(word)]), 4)})
        return ([[0.0, 0.0, float(w), float(h)]] if top == word else []), raw


ADAPTERS = {"locate_anything": LocateAnything, "rex_omni": RexOmni, "gemini": Gemini, "gpt": OpenAIVLM,
            "gpt_astra": GPT6Astra, "clip": CLIPClassifier}


# --------------------------------------------------------------------- helpers
def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# the 8-class map the v1 renders were made with (before the switch to 12 categories)
V1_FALSE_CATEGORY_MAP = {"chair": "sofa", "sofa": "chair", "cabinet": "bookshelf", "bookshelf": "cabinet",
                         "bed": "sofa", "table": "cabinet", "lamp": "bookshelf", "bathtub": "bed"}


def false_prompts(choice="config"):
    if choice == "v1":
        return V1_FALSE_CATEGORY_MAP, {}       # prompts fall back to "a <word>", as in v1
    if choice != "config":
        return json.load(open(choice)), {}
    try:
        sys.path.insert(0, str(ROOT / "rendering"))
        from config import CATEGORY_PROMPTS, FALSE_CATEGORY_MAP
        return FALSE_CATEGORY_MAP, CATEGORY_PROMPTS
    except Exception as e:
        sys.exit(f"--false-positive needs rendering/config.py (FALSE_CATEGORY_MAP): {e}")


def default_output(manifest, model, false_mode):
    name = manifest.name
    if name == "manifest.jsonl":
        stem = "false_positive_inference_log" if false_mode else "inference_log"
    elif name == "framing_manifest.jsonl":
        stem = "framing_inference_log" if false_mode else "framing_true_inference_log"
    elif name == "query_manifest.jsonl":
        stem = "benchmark_inference_log"
    elif name == "orbit_plan.jsonl":
        stem = "orbit_inference_log" if false_mode else "orbit_true_inference_log"
    else:
        stem = manifest.stem + ("_false" if false_mode else "") + "_inference_log"
    return manifest.parent / f"{stem}_{model}.jsonl"


def resolve_image(rgb_path, manifest):
    """rgb_path is stored relative to viewpoint-sensitivity/ (data/renders/...). If the data was
    moved (e.g. into v1/), fall back to the same trailing path under the manifest's folder."""
    p = Path(rgb_path)
    if p.is_absolute():
        return p
    if (ROOT / p).exists():
        return ROOT / p
    for k in range(len(p.parts)):
        cand = manifest.parent / Path(*p.parts[k:])
        if cand.exists():
            return cand
    return ROOT / p          # not found: let the caller report the original path


def subsample(rows, per_cat):
    keep = {}
    for r in rows:
        ids = keep.setdefault(r["category"], [])
        if r["model_id"] not in ids and len(ids) < per_cat:
            ids.append(r["model_id"])
    allowed = {i for ids in keep.values() for i in ids}
    return [r for r in rows if r["model_id"] in allowed]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--model-name", default=None, help="override the HF id / API model name")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--false-positive", action="store_true",
                    help="prompt the FALSE_CATEGORY_MAP word instead of the real object (FP/TN pass)")
    ap.add_argument("--false-map", default="config",
                    help="wrong-word map for --false-positive: 'config' (rendering/config.py, default), "
                         "'v1' (the old 8-class map, for renders made before the 12-class switch), "
                         "or a path to a JSON {category: wrong_word}")
    ap.add_argument("--instances-per-category", type=int, default=0, help="subsample objects (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    ap.add_argument("--show-raw", action="store_true", help="print each raw model response")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reasoning-effort", default=None,
                    help="gpt_astra / gpt: low|medium|high|xhigh|max (gpt_astra defaults to low)")
    ap.add_argument("--no-env-fix", action="store_true",
                    help="LocateAnything: only REPORT an incompatible transformers version instead of "
                         "installing the version the model was saved with into the active conda env")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (API rate limits)")
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
    if not manifest.exists():
        found = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("**/manifest.jsonl"))[:10]
        sys.exit(f"manifest not found: {manifest}\nmanifests under {ROOT}:\n  " + "\n  ".join(found or ["(none)"]))
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    all_categories = sorted({r["category"] for r in rows})        # CLIP's label set: whole manifest
    if args.instances_per_category:
        rows = subsample(rows, args.instances_per_category)
    if args.limit:
        rows = rows[:args.limit]
    out = args.output or default_output(manifest, args.model, args.false_positive)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            if l.strip():
                r = json.loads(l)
                if r.get("error") is None:
                    done.add(r["view_id"])
    todo = [r for r in rows if r["view_id"] not in done]
    print(f"{args.model}: {len(rows)} rows, {len(done)} already done, {len(todo)} to run -> {out}")
    if not todo:
        return
    fmap, cprompts = false_prompts(args.false_map) if args.false_positive else ({}, {})
    if args.false_positive:
        unmapped = sorted({r["category"] for r in todo} - set(fmap))
        if unmapped:
            sys.exit(f"the wrong-word map ({args.false_map}) has no entry for {unmapped}. These renders were "
                     "made with a different category list: for the old renders use --false-map v1.")

    global NO_ENV_FIX
    NO_ENV_FIX = args.no_env_fix
    kw = {"device": args.device}
    if args.model_name:
        kw["model_name"] = args.model_name
    if args.reasoning_effort:
        kw["reasoning_effort"] = args.reasoning_effort
    if args.model == "clip":
        kw["labels"] = all_categories
        missing = sorted(set(fmap.values()) - set(all_categories)) if args.false_positive else []
        if missing:
            sys.exit(f"CLIP: wrong words {missing} are not object categories in this manifest, so CLIP "
                     "cannot pick them; use a wrong-word map within the category list.")
    adapter = ADAPTERS[args.model](**kw)

    t_start = time.time()
    with open(out, "a") as f:
        for i, r in enumerate(todo, 1):
            img_path = resolve_image(r["rgb_path"], manifest)
            gt = r.get("gt_bbox_xyxy")
            if args.false_positive:
                false_cat = fmap[r["category"]]
                prompt = cprompts.get(false_cat, f"a {false_cat}")
            else:
                prompt = r["prompt"]
            rec = {"view_id": r["view_id"], "model": args.model, "category": r["category"],
                   "model_id": r["model_id"], "prompt": prompt, "threshold": None}
            rec.update({k: r[k] for k in PASSTHROUGH if k in r})
            t0 = time.time()
            try:
                pil = Image.open(img_path).convert("RGB")
                boxes, raw = detect_with_retry(adapter, pil, prompt)
                best = boxes[0] if boxes else None
                rec.update({"num_detections": len(boxes), "best_score": 1.0 if best else None,
                            "best_box_xyxy": best, "error": None, "raw_response": str(raw)[:500]})
                ov = iou(best, gt) if gt else 0.0
                if args.false_positive:
                    rec.update({"true_category": r["category"], "false_category": false_cat,
                                "false_positive": best is not None,
                                "iou_with_true_object": ov if best else None,
                                "relabeled_true_object": bool(best is not None and ov >= 0.5)})
                else:
                    rec.update({"iou_with_gt": ov, "correct": bool(ov >= 0.5)})
                if args.show_raw:
                    print(f"--- {r['view_id']} | {prompt!r}\n{str(raw)[:400]}\n-> boxes {boxes} | IoU {ov:.2f}")
            except Exception as e:
                rec.update({"num_detections": 0, "best_score": None, "best_box_xyxy": None,
                            "error": f"{type(e).__name__}: {e}"})
                print(f"  error on {r['view_id']}: {rec['error'][:300]}")
                if AUTH_ERROR.search(rec["error"]):
                    f.write(json.dumps(rec) + "\n")
                    credits = re.search(r"insufficient_quota|credit_balance_exhausted|billing|exceeded your current quota",
                                        rec["error"], re.I)
                    sys.exit(AUTH_HELP["credits"] if credits else
                             AUTH_HELP.get(args.model, "authentication failed: check the API key / login for this model"))
            rec["latency_ms"] = 1000 * (time.time() - t0)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t_start)
                print(f"  {i}/{len(todo)}  ({rate:.2f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left)")
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
