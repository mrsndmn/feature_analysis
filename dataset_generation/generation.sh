#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

models=(
    "google/siglip-base-patch16-224"
    "google/siglip-base-patch16-256"
    "google/siglip-base-patch16-384"
    "google/siglip-base-patch16-512"
    "google/siglip2-base-patch16-224"
    "google/siglip2-base-patch16-256"
    "google/siglip2-base-patch16-384"
    "google/siglip2-base-patch16-512"
)

coco_images_path="$SCRIPT_DIR/../coco_subsets/val2017"
split="val"
max_count=10

for model in "${models[@]}"; do
    echo "========================================"
    echo "Processing model: $model"
    echo "========================================"
    
    python "$SCRIPT_DIR/generation.py" \
        --vision_model_name "$model" \
        --coco_images_path "$coco_images_path" \
        --split "$split" \
        --max_count "$max_count"
    
    echo "Finished processing model: $model"
    echo
done

echo "All models processed successfully!"