#!/usr/bin/env bash
# Downloads ShapeNetCore .glb models for the categories in rendering/config.py
# (SHAPENET_SYNSETS) into ./data/shapenet/<synset_id>/*.glb.
#
# Source: https://huggingface.co/datasets/ShapeNet/shapenetcore-glb -- a gated
# dataset. Before this will work you must:
#   1. Log into huggingface.co and accept the ShapeNet research-use license on
#      that dataset's page.
#   2. Create a read-access token at https://huggingface.co/settings/tokens
#      and export it: `export HF_TOKEN=hf_...`
#
# Requires `huggingface_hub` (pip install huggingface_hub) -- this can run in
# any env with network access, it does not need habitat-sim.
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./scripts/download_shapenet.sh                 # all categories in config.py
#   ./scripts/download_shapenet.sh chair sofa       # just these categories
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Accept the license at " \
       "https://huggingface.co/datasets/ShapeNet/shapenetcore-glb , create a " \
       "token at https://huggingface.co/settings/tokens, and export HF_TOKEN=hf_..." >&2
  exit 1
fi

if ! command -v huggingface-cli &>/dev/null; then
  echo "huggingface-cli not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

DEST="$(pwd)/data/shapenet"
mkdir -p "$DEST"

# category -> synset id, kept in sync with rendering/config.py:SHAPENET_SYNSETS
declare -A SYNSETS=(
  [chair]=03001627
  [table]=04379243
  [sofa]=04256520
  [cabinet]=02933112
  [bed]=02818832
  [bookshelf]=02871439
  [lamp]=03636649
  [bathtub]=02808440
)

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=("${!SYNSETS[@]}")
fi

for category in "${TARGETS[@]}"; do
  synset="${SYNSETS[$category]:-}"
  if [[ -z "$synset" ]]; then
    echo "Unknown category '$category' (not in rendering/config.py:SHAPENET_SYNSETS)" >&2
    exit 1
  fi
  echo "== downloading $category (synset $synset) =="
  # NOTE: `--include` pattern below assumes files live under a top-level
  # "<synset_id>/" prefix in the HF repo. Verify against the actual repo file
  # listing (huggingface-cli repo files ShapeNet/shapenetcore-glb --repo-type
  # dataset) and adjust the pattern if the real layout differs.
  huggingface-cli download ShapeNet/shapenetcore-glb \
    --repo-type dataset \
    --include "${synset}/*" \
    --local-dir "$DEST" \
    --token "$HF_TOKEN"
done

echo
echo "Done. Verify with: find data/shapenet -name '*.glb' | wc -l"
