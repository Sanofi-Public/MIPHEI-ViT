set -ex
python train.py \
    --dataroot /root/workdir/ORION_dataset_20x \
    --dataset_mode orion \
    --name orion_pix2pix_v2 \
    --model pix2pix \
    --netG unet_256 \
    --direction AtoB \
    --lambda_L1 30 \
    --batch_size 2 \
    --lr 0.00003 \
    --lr_policy step \
    --norm batch \
    --preprocess crop \
    --pool_size 0 \
    --output_nc 16 \
    --load_size 333 \
    --crop_size 256 \
    --n_epochs 10 \
    --n_epochs_decay 6 \
    --save_epoch_freq 1
```
