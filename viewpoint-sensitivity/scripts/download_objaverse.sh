#!/usr/bin/env bash
# Downloads Objaverse (LVIS-annotated) objects for the categories in
# rendering/config.py:OBJAVERSE_CATEGORY_KEYWORDS. No HF login/license needed --
# this is the dataset to use while ShapeNet's gated-license approval is pending
# (see rendering/config.py:DATASET_SOURCE).
#
# The actual .glb bytes land in the `objaverse` package's own cache dir
# (~/.objaverse/hf-objaverse-v1/glbs/...), NOT under this repo's data/ --
# only a small manifest (data/objaverse/category_uid_map.json) is written here.
#
# Usage:
#   conda activate vps-render
#   ./scripts/download_objaverse.sh --dry-run   # show matched LVIS keys + counts, download nothing
#   ./scripts/download_objaverse.sh              # actually download
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python rendering/download_objaverse.py "$@"
