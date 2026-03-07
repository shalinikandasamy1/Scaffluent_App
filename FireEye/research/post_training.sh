#!/bin/bash
# Post-training automation: runs after YOLO training completes on .153
# Evaluates model, generates missing data, launches round 2
# Run from: FireEye/research/ with forge venv activated

set -e

VENV="/home/evnchn/pinokio/api/stable-diffusion-webui-forge.git/app/venv"
source "$VENV/bin/activate"

cd /home/evnchn/Scaffluent_App/FireEye/research

BEST_WEIGHTS="/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run1/weights/best.pt"

echo "=== Post-Training Pipeline ==="
echo "Time: $(date '+%H:%M %Z')"
echo ""

# Step 1: Evaluate on real images
echo "=== Step 1: Evaluate run 1 on real images ==="
python3 evaluate_model.py \
    --model "$BEST_WEIGHTS" \
    --images "/home/evnchn/Scaffluent_App/Images dataset/Real/" \
    --output ./eval_run1 \
    --conf 0.15

# Post e-paper update
python3 -c "
import json, urllib.request, csv
rows = list(csv.DictReader(open('/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run1/results.csv')))
best = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
data = json.dumps({'header': 'Run 1 Done!', 'body': f\"mAP50={float(best['metrics/mAP50(B)']):.3f} @ep{best['epoch']}\nEvaluating real images...\"}).encode()
req = urllib.request.Request('http://192.168.50.72:8090/api/message', data=data, headers={'Content-Type': 'application/json'})
urllib.request.urlopen(req, timeout=5)
"

# Step 2: Generate fire extinguisher data
echo ""
echo "=== Step 2: Generate fire extinguisher images ==="
python3 generate_fire_extinguisher_data.py \
    --output ./extinguisher_data \
    --num-per-prompt 3 \
    --threshold 0.15

# Step 3: Auto-label welding frames
echo ""
echo "=== Step 3: Auto-label welding frames ==="
python3 auto_label_v2.py \
    --images ./welding_frames \
    --labels ./welding_labels \
    --threshold 0.18

# Step 4: Create ground truth for real images
echo ""
echo "=== Step 4: Create real image ground truth ==="
python3 create_real_gt.py \
    --images "/home/evnchn/Scaffluent_App/Images dataset/Real/" \
    --output ./real_image_gt \
    --threshold 0.20

# Step 5: Merge v2 dataset
echo ""
echo "=== Step 5: Build merged dataset v2 ==="
python3 merge_datasets_v2.py --output ./merged_dataset_v2

# Step 6: Clean labels
echo ""
echo "=== Step 6: Clean noisy labels ==="
python3 clean_labels.py --dir ./merged_dataset_v2

# Step 7: Train round 2
echo ""
echo "=== Step 7: Train round 2 (from run 1 best weights) ==="
python3 train_yolo_fireeye.py \
    --data ./merged_dataset_v2/dataset.yaml \
    --epochs 60 \
    --batch 16 \
    --model "$BEST_WEIGHTS" \
    --project ./yolo_finetune \
    --name merged_run2 \
    --patience 20

# Step 8: Final evaluation
echo ""
echo "=== Step 8: Evaluate round 2 on real images ==="
python3 evaluate_model.py \
    --model /home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run2/weights/best.pt \
    --images "/home/evnchn/Scaffluent_App/Images dataset/Real/" \
    --output ./eval_run2 \
    --conf 0.15

# Step 9: Compare runs
echo ""
echo "=== Step 9: Compare all runs ==="
python3 compare_runs.py

# Final e-paper update
python3 -c "
import json, urllib.request
data = json.dumps({'header': 'All Training Done!', 'body': 'Run1 + Run2 complete\nCheck eval_run1/ eval_run2/'}).encode()
req = urllib.request.Request('http://192.168.50.72:8090/api/message', data=data, headers={'Content-Type': 'application/json'})
urllib.request.urlopen(req, timeout=5)
"

echo ""
echo "=== Pipeline complete! ==="
echo "Results:"
echo "  Run 1 eval: ./eval_run1/"
echo "  Run 2 eval: ./eval_run2/"
echo "  Best weights: check compare_runs.py output"
