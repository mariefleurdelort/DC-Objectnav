#!/usr/bin/env bash
# 12-word benchmark. Usage: BENCH_ROOT=v1 bash run_bench.sh servers clip locate_anything rex_omni
set -u
export PYTHONNOUSERSITE=1
ROOT_DIR="${BENCH_ROOT:-.}"
Q="$ROOT_DIR/data/renders_benchmark/query_manifest.jsonl"
[ -f "$Q" ] || { echo "missing $Q: run expand_manifest_queries.py first"; exit 1; }
declare -A ENV=( [locate_anything]=vps-locate [clip]=vps-locate [rex_omni]=vps-rex [gpt_astra]=vps-rex [gemini]=vps-rex )
for M in "$@"; do
  if [[ "$M" == "servers" ]]; then
    echo "=== server models ==="
    conda run --no-capture-output -n vps-orchestrator python orchestrator/run_inference.py \
        --models groundingdino owlv2 sam3 yoloworld \
        --manifest "$Q" --output "$ROOT_DIR/data/renders_benchmark/benchmark_inference_log.jsonl"
  else
    echo "=== $M (env ${ENV[$M]}) ==="
    conda run --no-capture-output -n "${ENV[$M]}" python run_vlm_inference.py --model "$M" --manifest "$Q" \
        || echo "!!! $M failed (see above)"
  fi
done
echo "=== score ==="
conda run --no-capture-output -n vps-orchestrator python results_table.py --root "$ROOT_DIR" --experiment bench \
    --models sam3 clip yoloworld owlv2 locate_anything rex_omni groundingdino
conda run --no-capture-output -n vps-orchestrator python fp_resolution.py --root "$ROOT_DIR" --experiment bench
