#!/usr/bin/env bash
# Run LocateAnything / Rex-Omni / CLIP / GPT-6 Astra / Gemini / GPT on the v1 main-sweep renders (right word + wrong
# word), each in its own conda env, then score all models together. Resumable: re-run the
# script after any interruption and finished rows are skipped.
#
#   cd ~/projects/DC-ObjectNav/viewpoint-sensitivity
#   tmux new -s vlm
#   bash run_all_vlms.sh                       # all models listed in MODELS below
#   bash run_all_vlms.sh locate_anything       # just one
#   bash run_all_vlms.sh gpt_astra             # GPT-6 Astra (needs OPENAI_API_KEY)
#   (detach: Ctrl+B then D; reattach: tmux attach -t vlm)
#
# GPT-6 Astra / Gemini / GPT are paid APIs: export OPENAI_API_KEY / GEMINI_API_KEY first, and keep
# API_SUBSAMPLE small until you know the cost.
set -u
MANIFEST="v1/data/renders/manifest.jsonl"
API_SUBSAMPLE=5                      # objects per category for the paid APIs (0 = all)

declare -A ENV=( [locate_anything]=vps-locate [rex_omni]=vps-rex [gemini]=vps-rex [gpt]=vps-rex [gpt_astra]=vps-rex [clip]=vps-locate )
MODELS=("${@:-locate_anything rex_omni clip}")
MODELS=(${MODELS[@]})

for M in "${MODELS[@]}"; do
  EXTRA=""
  if [[ "$M" == "gemini" || "$M" == "gpt" || "$M" == "gpt_astra" ]]; then
    EXTRA="--instances-per-category $API_SUBSAMPLE --sleep 0.3"
  fi
  echo "=== $M (env ${ENV[$M]}): right-word pass ==="
  conda run --no-capture-output -n "${ENV[$M]}" \
      python run_vlm_inference.py --model "$M" --manifest "$MANIFEST" $EXTRA \
      || { echo "!!! $M right-word pass failed, skipping $M"; continue; }
  echo "=== $M: wrong-word pass ==="
  conda run --no-capture-output -n "${ENV[$M]}" \
      python run_vlm_inference.py --model "$M" --manifest "$MANIFEST" --false-positive --false-map v1 $EXTRA \
      || echo "!!! $M wrong-word pass failed"
done

echo "=== scoring all models ==="
conda run --no-capture-output -n vps-orchestrator python confusion_metrics.py --root v1 2>&1 \
    | tee v1/confusion_summary_all_models.txt
