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
# PanNuke
# ---------------------------------------------------------
echo "[PanNuke] Downloading..."
mkdir -p "$DATA_DIR/pannuke/orig_data"
cd "$DATA_DIR/pannuke/orig_data"

for i in 1 2 3; do
    aria2c -x 16 -s 16 -k 10M \
        -o fold_$i.zip \
        "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_$i.zip" && \
        unzip -q fold_$i.zip && \
        rm fold_$i.zip
done

cd - >/dev/null

echo "----------------------------------------"
echo "PanNuke dataset downloaded into: $DATA_DIR/pannuke/orig_data"
echo "----------------------------------------"
