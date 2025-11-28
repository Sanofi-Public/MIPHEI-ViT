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
# ORIONCRC
# ---------------------------------------------------------
echo "[ORIONCRC] Downloading..."
cd "$DATA_DIR"

aria2c -x 16 -s 16 -k 10M \
  -x 16 -s 16 -k 10M \
  --timeout=60 \
  --connect-timeout=60 \
  --max-connection-per-server=4 \
  --max-tries=15 \
  --retry-wait=5 \
  --continue=true \
  --check-certificate=false \
  -o ORIONCRC_dataset_tile_20x.zip \
  "https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1" && \
    unzip -q ORIONCRC_dataset_tile_20x.zip && \
    rm ORIONCRC_dataset_tile_20x.zip

curl -L \
  -o ORIONCRC_dataset_tile_20x/val_test_nuclei_dataframe.parquet \
  "https://drive.usercontent.google.com/download?id=1XbZX3dlfzBZTVDOnbPYRzbCrs5xX2sMX&export=download"

cd - >/dev/null

echo "----------------------------------------"
echo "OrionCRC dataset downloaded into: $DATA_DIR/ORION_dataset_20x"
echo "----------------------------------------"