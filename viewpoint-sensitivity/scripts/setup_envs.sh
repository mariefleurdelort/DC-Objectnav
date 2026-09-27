#!/usr/bin/env bash
# Creates all isolated conda environments for the viewpoint-sensitivity pipeline:
#   vps-render        - habitat-sim rendering env
#   vps-yoloworld      \
#   vps-groundingdino   } 4 isolated model-server envs
#   vps-sam3            /
#   vps-owlv2          /
#   vps-orchestrator  - thin env that only talks HTTP to the servers above
#
# Usage:
#   ./scripts/setup_envs.sh            # create all envs
#   ./scripts/setup_envs.sh render     # create just one (name after the .yml stem)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_DIR="environment"
ALL_ENVS=(render_habitat server_yoloworld server_groundingdino server_sam3 server_owlv2 orchestrator)

if ! command -v conda &>/dev/null; then
  echo "conda not found on PATH. Install Miniconda/Miniforge first: https://github.com/conda-forge/miniforge" >&2
  exit 1
fi

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=("${ALL_ENVS[@]}")
fi

for target in "${TARGETS[@]}"; do
  yml="$ENV_DIR/${target}.yml"
  if [ ! -f "$yml" ]; then
    echo "No such env spec: $yml" >&2
    exit 1
  fi
  env_name=$(grep -m1 '^name:' "$yml" | awk '{print $2}')
  if conda env list | grep -qE "^${env_name}\s"; then
    echo "== $env_name already exists, updating in place =="
    conda env update -n "$env_name" -f "$yml" --prune
  else
    echo "== creating $env_name from $yml =="
    conda env create -f "$yml"
  fi
done

echo
echo "Done. Verify with: conda env list"
