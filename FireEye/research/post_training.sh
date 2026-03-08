#!/bin/bash
# Run after training completes to compare and optionally deploy.
# Usage: bash research/post_training.sh

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Run 5 Training Complete ==="
echo ""

# Show final metrics
tail -1 /home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run5/results.csv | \
    awk -F',' '{printf "Final epoch %s: mAP50=%.3f, mAP50-95=%.3f\n", $1, $8, $9}'

echo ""
echo "=== Per-Class Comparison ==="
python research/compare_per_class.py run4 run5

echo ""
echo "=== Evaluate Run 5 on test images ==="
BEST_PT="/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run5/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    python evaluate.py --heuristic-only --model "$BEST_PT" --save
else
    echo "WARNING: best.pt not found at $BEST_PT"
fi

echo ""
echo "To deploy Run 5:"
echo "  python research/deploy_model.py run5 --name fireeye_yolo11n_v5.pt"
