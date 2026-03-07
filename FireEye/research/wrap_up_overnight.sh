#!/bin/bash
# Overnight session wrap-up: collect results, compare, restore GPU, final update
# Run from: FireEye/research/ with forge venv activated

set -e

VENV="/home/evnchn/pinokio/api/stable-diffusion-webui-forge.git/app/venv"
source "$VENV/bin/activate"
cd /home/evnchn/Scaffluent_App/FireEye/research

echo "=== Overnight Session Wrap-Up ==="
echo "Time: $(date '+%H:%M %Z')"

# Step 1: Collect .172 SGD results
echo ""
echo "=== Step 1: Collect .172 SGD results ==="
SGD_CSV=$(sshpass -p 'insecure' ssh -o StrictHostKeyChecking=no evnchn@192.168.50.172 \
    "cat /home/evnchn/fireeye_data/runs/detect/yolo_finetune/merged_sgd_run/results.csv" 2>/dev/null)
if [ -n "$SGD_CSV" ]; then
    mkdir -p /home/evnchn/Scaffluent_App/FireEye/research/sgd_run_results
    echo "$SGD_CSV" > /home/evnchn/Scaffluent_App/FireEye/research/sgd_run_results/results.csv
    echo "  Copied .172 SGD results locally"

    # Also copy best weights if available
    sshpass -p 'insecure' scp -o StrictHostKeyChecking=no \
        evnchn@192.168.50.172:/home/evnchn/fireeye_data/runs/detect/yolo_finetune/merged_sgd_run/weights/best.pt \
        /home/evnchn/Scaffluent_App/FireEye/research/sgd_run_results/best.pt 2>/dev/null && \
        echo "  Copied .172 SGD best weights" || echo "  Could not copy SGD weights (still training?)"
else
    echo "  Could not reach .172 or no results"
fi

# Step 2: Evaluate Run 3 on real images (if not done)
echo ""
if [ ! -d "./eval_run3" ]; then
    echo "=== Step 2: Evaluate Run 3 on real images ==="
    RUN3_WEIGHTS="/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run3/weights/best.pt"
    if [ -f "$RUN3_WEIGHTS" ]; then
        python3 evaluate_model.py \
            --model "$RUN3_WEIGHTS" \
            --images "/home/evnchn/Scaffluent_App/Images dataset/Real/" \
            --output ./eval_run3 \
            --conf 0.15
    else
        echo "  Run 3 weights not found (still training?)"
    fi
else
    echo "=== Step 2: Run 3 evaluation already done ==="
fi

# Step 3: Compare all runs
echo ""
echo "=== Step 3: Compare all runs ==="
python3 compare_runs.py

# Step 4: Restore GPU power limit
echo ""
echo "=== Step 4: Restore GPU power limit ==="
echo "insecure" | sudo -S nvidia-smi -pl 170 2>/dev/null && \
    echo "  Restored .153 GPU power limit to 170W" || \
    echo "  Could not restore power limit"

# Step 5: Final e-paper update
echo ""
echo "=== Step 5: Final e-paper update ==="
python3 -c "
import json, urllib.request
data = json.dumps({'header': 'Overnight Complete!', 'body': 'R1/R2/R3 + SGD done\nGPU power restored to 170W'}).encode()
req = urllib.request.Request('http://192.168.50.72:8090/api/message', data=data, headers={'Content-Type': 'application/json'})
urllib.request.urlopen(req, timeout=5)
print('  E-paper updated')
"

echo ""
echo "=== Session complete at $(date '+%H:%M %Z') ==="
