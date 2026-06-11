#!/bin/bash
# Downloads SMPL GLB body models from HuggingFace into motion_viewer/public/models/

set -e

MODELS_DIR="motion_viewer/public/models"
BASE_URL="https://huggingface.co/datasets/anukaishara123/ipcv-motion-viewer-assets/resolve/main"

mkdir -p "$MODELS_DIR"

echo "Downloading SMPL body models..."

for model in smpl_female.glb smpl_male.glb; do
    if [ -f "$MODELS_DIR/$model" ]; then
        echo "  $model already exists, skipping."
    else
        echo "  Downloading $model..."
        curl -L --progress-bar "$BASE_URL/$model" -o "$MODELS_DIR/$model"
    fi
done

echo "Done. Models saved to $MODELS_DIR/"
