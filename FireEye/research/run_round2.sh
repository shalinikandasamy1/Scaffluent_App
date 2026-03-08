#!/bin/bash
# Round 2 training pipeline: improved data quality
# Run after round 1 completes and GPU is free
# From: FireEye/research/ with forge venv activated

set -e

VENV="/home/evnchn/pinokio/api/stable-diffusion-webui-forge.git/app/venv"
source "$VENV/bin/activate"

echo "=== Round 2: Improved FireEye Training ==="
echo ""

# Step 1: Generate fire extinguisher data (fixes the ZERO class gap)
echo "=== Step 1: Generate fire extinguisher images ==="
python3 generate_fire_extinguisher_data.py \
    --output ./extinguisher_data \
    --num-per-prompt 3 \
    --threshold 0.15

# Step 2: Auto-label welding frames
echo ""
echo "=== Step 2: Auto-label welding frames ==="
python3 auto_label_v2.py \
    --images ./welding_frames \
    --labels ./welding_labels \
    --threshold 0.18

# Step 3: Clean and re-merge dataset
echo ""
echo "=== Step 3: Re-merge with cleaned labels ==="
# Back up original merged dataset
if [ -d ./merged_dataset_v2 ]; then
    rm -rf ./merged_dataset_v2
fi
python3 merge_datasets_v2.py --output ./merged_dataset_v2

# Step 4: Clean labels
echo ""
echo "=== Step 4: Clean noisy labels ==="
python3 clean_labels.py --dir ./merged_dataset_v2

# Step 5: Train round 2 (start from round 1 best weights for faster convergence)
echo ""
echo "=== Step 5: Train YOLO round 2 ==="
BEST_WEIGHTS="$(ls /home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run1/weights/best.pt 2>/dev/null || echo /home/evnchn/Scaffluent_App/yolo11n.pt)"
echo "Starting from: $BEST_WEIGHTS"

python3 train_yolo_fireeye.py \
    --data ./merged_dataset_v2/dataset.yaml \
    --epochs 80 \
    --batch 16 \
    --model "$BEST_WEIGHTS" \
    --project ./yolo_finetune \
    --name merged_run2 \
    --patience 20

# Step 6: Evaluate on real images
echo ""
echo "=== Step 6: Evaluate on real images ==="
python3 evaluate_model.py \
    --model /home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run2/weights/best.pt \
    --images "/home/evnchn/Scaffluent_App/Images dataset/Real/" \
    --output ./eval_round2

echo ""
echo "=== Round 2 complete! ==="
