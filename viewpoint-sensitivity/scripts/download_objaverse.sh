#!/usr/bin/env bash
# Downloads Objaverse (LVIS-annotated) objects 

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python rendering/download_objaverse.py "$@"
