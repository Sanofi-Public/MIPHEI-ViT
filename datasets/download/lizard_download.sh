#!/bin/bash
set -e

# ------------------------------
# CHECK ARGUMENT
# ------------------------------
if [ -z "$1" ]; then
    echo "Usage: $0 <target_data_folder>"
    exit 1
fi

DATA_DIR="$1"

# Normalize to absolute path
DATA_DIR=$(realpath "$DATA_DIR")

mkdir -p "$DATA_DIR"

echo "Downloading all datasets into: $DATA_DIR"
echo "----------------------------------------"

# ---------------------------------------------------------
# Lizard
# ---------------------------------------------------------
echo "[Lizard] Downloading..."
mkdir -p "$DATA_DIR/lizard"
cd "$DATA_DIR/lizard"

kaggle datasets download -d aadimator/lizard-dataset -p ./ --unzip

cd - >/dev/null

echo "----------------------------------------"
echo "Lizard dataset downloaded into: $DATA_DIR/lizard"
echo "----------------------------------------"