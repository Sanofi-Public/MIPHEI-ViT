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
# PathoCell (huggingface)
# ---------------------------------------------------------
echo "[PathoCell] Downloading..."
mkdir -p "$DATA_DIR/pathocell"
cd "$DATA_DIR/pathocell"

hf download \
  Kainmueller-Lab/PathoCell \
  --repo-type dataset \
  --include "pathocell_hdf/**" \
  --local-dir ./

cd - >/dev/null

echo "----------------------------------------"
echo "PathoCell dataset downloaded into: $DATA_DIR/pathocell"
echo "----------------------------------------"