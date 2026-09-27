# run_scene.sh
#!/bin/bash
set -euo pipefail
SCENE=$1
DATA_PATH="data/datasets/objectnav/gibson/dcon_subset/v1.1_sub10/val/content_fixed/${SCENE}.json.gz"

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: $DATA_PATH does not exist" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/$(date +%Y%m%d_%H%M%S)_${SCENE}.log"

CUDA_VISIBLE_DEVICES=3 python eval_with_trajectory.py \
  habitat_baselines.evaluate=True \
  habitat_baselines.test_episode_count=-1 \
  habitat.dataset.data_path="$DATA_PATH" \
  habitat.dataset.content_scenes=["$SCENE"] \
  habitat_baselines.num_environments=1 \
  2>&1 | tee "$LOG"