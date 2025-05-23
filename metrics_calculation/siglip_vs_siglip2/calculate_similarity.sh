#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
models_and_weights=(
    "google/siglip-base-patch16-224:models--google--siglip-base-patch16-224.pt"
    "google/siglip-base-patch16-256:models--google--siglip-base-patch16-256.pt"
    "google/siglip-base-patch16-384:models--google--siglip-base-patch16-384.pt"
    "google/siglip-base-patch16-512:models--google--siglip-base-patch16-512.pt"
    "google/siglip2-base-patch16-224:models--google--siglip2-base-patch16-224.pt"
    "google/siglip2-base-patch16-256:models--google--siglip2-base-patch16-256.pt"
    "google/siglip2-base-patch16-384:models--google--siglip2-base-patch16-384.pt"
    "google/siglip2-base-patch16-512:models--google--siglip2-base-patch16-512.pt"

)

max_count=10
for pair in "${models_and_weights[@]}"; do
    IFS=":" read -r model weights <<< "$pair"
    
    weights_path="$SCRIPT_DIR/../precalculated_weights/$weights"
    
    echo "========================================"
    echo "Processing model: $model"
    echo "Weights path: $weights_path"
    echo "========================================"
    
    python "$SCRIPT_DIR/calculate_similarity.py" \
        --vision_model_name "$model" \
        --max_count "$max_count" \
        --reconstructor_weights_path "$weights_path"
    
    echo "Finished processing model: $model"
    echo
done

echo "All models processed successfully!"