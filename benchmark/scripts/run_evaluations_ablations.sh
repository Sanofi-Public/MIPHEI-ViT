#!/bin/bash


# UNETR Hoptimus0
#python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/foundation_model_hoptimus --model miphei --dataset orion --min_area=10

# Pix2Pix UNETR Hoptimus0
#python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/full_gan --model miphei --dataset orion --min_area=10

# UNETR CTransPath
#python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/foundation_model_ctranspath --model miphei --dataset orion --min_area=10

# UNETR UNIv2
#python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/foundation_model_univ2 --model miphei --dataset orion --min_area=10

# UNETR Hoptimus0 Frozen
python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/unetr_hoptimus_frozen --model miphei --dataset orion --min_area=10

# ResNet50
#python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/resnet50 --model miphei --dataset orion --min_area=10

# MIPHEI HEMIT
python run_benchmark.py --checkpoint_dir /root/workdir/checkpoints_paper/hemit --model miphei --dataset orion --min_area=10

# MIPHEI ConvNeXt
#python run_benchmark.py --checkpoint_dir /root/workdir/MIPHEI-ViT/checkpoints/MIPHEI-convnext --model miphei --dataset orion --min_area=10

# MIPHEI
#python run_benchmark.py --checkpoint_dir /root/workdir/MIPHEI-ViT/checkpoints/MIPHEI-vit --model miphei --dataset orion --min_area=10
