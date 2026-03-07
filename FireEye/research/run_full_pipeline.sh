#!/bin/bash
# Full pipeline: auto-label welding frames → merge all datasets → train YOLO
# Run from: FireEye/research/ with forge venv activated

set -e

echo "=== Step 1: Auto-label welding frames ==="
python3 auto_label_v2.py \
    --images ./welding_frames \
    --labels ./welding_labels \
    --threshold 0.18

echo ""
echo "=== Step 2: Merge all datasets ==="
python3 merge_datasets.py --output ./merged_dataset

echo ""
echo "=== Step 3: Train YOLO on merged dataset ==="
python3 train_yolo_fireeye.py \
    --data ./merged_dataset/dataset.yaml \
    --epochs 50 \
    --batch 16 \
    --model /home/evnchn/Scaffluent_App/yolo11n.pt \
    --project ./yolo_finetune \
    --name merged_run1

echo ""
echo "=== Pipeline complete! ==="
