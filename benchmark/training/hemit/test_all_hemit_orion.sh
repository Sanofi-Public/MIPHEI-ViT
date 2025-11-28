#!/usr/bin/env bash

CHECKPOINT_DIR="checkpoints"
NAME="hemit_SwinTResnet_New_2"
DATAROOT="/root/workdir/ORION_dataset_20x"

CKPT_PATH="${CHECKPOINT_DIR}/${NAME}"

# Check existence
if [ ! -d "$CKPT_PATH" ]; then
    echo "Folder not found: $CKPT_PATH"
    exit 1
fi

# Loop over checkpoint files
for ckpt in $(ls "${CKPT_PATH}"/*_net_G.pth | sort -V); do
    filename=$(basename "$ckpt")
    epoch="${filename%%_net_G.pth}"

    echo ">> Running epoch $epoch"

    python test_metrics.py \
        --checkpoints_dir "$CHECKPOINT_DIR" \
        --epoch "$epoch" \
        --phase val \
        --dataroot "$DATAROOT" \
        --name "$NAME" \
        --model pix2pix \
        --eval \
        --netG SwinTResnet \
        --direction AtoB \
        --dataset_mode orion \
        --norm batch \
        --output_nc 16 \
        --load_size 333 \
        --crop_size 256 \
        --preprocess crop
done
