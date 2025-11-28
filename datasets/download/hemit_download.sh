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
# HEMIT
# ---------------------------------------------------------
echo "[HEMIT] Downloading..."
mkdir -p "$DATA_DIR/HEMIT_dataset"
cd "$DATA_DIR/HEMIT_dataset"

# ---------------------------------------
# Download original HEMIT dataset
# ---------------------------------------
aria2c -x 16 -s 16 -k 10M \
  -o HEMIT_raw.zip \
  "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/3gx53zm49d-1.zip" && \
    unzip -q HEMIT_raw.zip && \
    rm HEMIT_raw.zip

# Extract the .7z inside the extracted folder
7z x "HEMIT H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator/HEMIT.7z" && \
    rm -r "HEMIT H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator"

# ---------------------------------------
# Download nuclei analysis supplementary
# ---------------------------------------
aria2c -x 16 -s 16 -k 10M \
  -o HEMIT_nuclei_analysis.zip \
  "https://zenodo.org/records/15340874/files/HEMIT_nuclei_analysis.zip?download=1" && \
    unzip -q HEMIT_nuclei_analysis.zip && \
    rm HEMIT_nuclei_analysis.zip

# Move nuclei folders where they belong
for SPLIT in train val test; do
    if [ -d "HEMIT_nuclei_analysis/$SPLIT" ]; then
        mv "HEMIT_nuclei_analysis/$SPLIT"/* "$SPLIT/"
    fi
done

# Remove wrapper folder
rm -rf HEMIT_nuclei_analysis

cd - >/dev/null

echo "----------------------------------------"
echo "HEMIT dataset downloaded into: $DATA_DIR/HEMIT_dataset"
echo "----------------------------------------"