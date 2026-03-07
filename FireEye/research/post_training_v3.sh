#!/bin/bash
# Post-training pipeline for Run 3: evaluate, compare all runs, update report
# Run from: FireEye/research/ with forge venv activated

set -e

VENV="/home/evnchn/pinokio/api/stable-diffusion-webui-forge.git/app/venv"
source "$VENV/bin/activate"

cd /home/evnchn/Scaffluent_App/FireEye/research

RUN3_WEIGHTS="/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run3/weights/best.pt"

echo "=== Post-Training Pipeline v3 ==="
echo "Time: $(date '+%H:%M %Z')"
echo ""

# Step 1: Evaluate Run 3 on real images
echo "=== Step 1: Evaluate Run 3 on real images ==="
python3 evaluate_model.py \
    --model "$RUN3_WEIGHTS" \
    --images "/home/evnchn/Scaffluent_App/Images dataset/Real/" \
    --output ./eval_run3 \
    --conf 0.15

# Step 2: Compare all runs
echo ""
echo "=== Step 2: Compare all runs ==="
python3 compare_runs.py

# Step 3: E-paper update
python3 -c "
import json, urllib.request, csv
rows = list(csv.DictReader(open('/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run3/results.csv')))
best = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
data = json.dumps({'header': 'Run 3 Complete!', 'body': f\"mAP50={float(best['metrics/mAP50(B)']):.3f} @ep{best['epoch']}\nEval done, check results\"}).encode()
req = urllib.request.Request('http://192.168.50.72:8090/api/message', data=data, headers={'Content-Type': 'application/json'})
urllib.request.urlopen(req, timeout=5)
"

echo ""
echo "=== Pipeline complete ==="
echo "Results:"
echo "  Run 3 eval: ./eval_run3/"
echo "  Comparison: see compare_runs.py output above"
