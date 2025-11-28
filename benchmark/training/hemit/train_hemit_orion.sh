set -ex
python train.py \
    --dataroot /root/workdir/ORION_dataset_20x \
    --dataset_mode orion \
    --name hemit_SwinTResnet_New_2 \
    --model pix2pix \
    --netG SwinTResnet \
    --direction AtoB \
    --display_id 0 \
    --lambda_L1 30 \
    --lr 0.00003 \
    --lr_policy step \
    --batch_size 2 \
    --loss_type L1 \
    --norm batch \
    --preprocess crop \
    --pool_size 0 \
    --output_nc 16 \
    --val_freq 5 \
    --load_size 333 \
    --crop_size 256 \
    --n_epochs 10 \
    --n_epochs_decay 6 \
    --save_epoch_freq 1
```
