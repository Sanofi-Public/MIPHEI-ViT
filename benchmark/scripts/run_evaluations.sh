#!/bin/bash


# Pix2Pix
python run_benchmark.py --checkpoint_dir checkpoints/pix2pix/ --model pix2pix --dataset orion --min_area=10
python run_benchmark.py --checkpoint_dir checkpoints/pix2pix/ --model pix2pix --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/pix2pix/ --model pix2pix --dataset hemit --min_area=0
python run_benchmark.py --checkpoint_dir checkpoints/pix2pix/ --model pix2pix --dataset lizard --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/pix2pix/ --model pix2pix --dataset pannuke --min_area=0

# HEMIT
python run_benchmark.py --checkpoint_dir checkpoints/HEMIT/ --model hemit --dataset orion --min_area=10 --batch_size 4
python run_benchmark.py --checkpoint_dir checkpoints/HEMIT/ --model hemit --dataset pathocell --min_area=0 --num_workers 0 --batch_size 4
python run_benchmark.py --checkpoint_dir checkpoints/HEMIT/ --model hemit --dataset hemit --min_area=0 --batch_size 2
python run_benchmark.py --checkpoint_dir checkpoints/HEMIT/ --model hemit --dataset lizard --min_area=0 --num_workers 0 --batch_size 4
python run_benchmark.py --checkpoint_dir checkpoints/HEMIT/ --model hemit --dataset pannuke --min_area=0 --batch_size 4

# MIPHEI_convnext
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-convnext/ --model miphei --dataset orion --min_area=10
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-convnext/ --model miphei --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-convnext/ --model miphei --dataset hemit --min_area=0
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-convnext/ --model miphei --dataset lizard --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-convnext/ --model miphei --dataset pannuke --min_area=0

# MIPHEI
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-vit/ --model miphei --dataset orion --min_area=10
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-vit/ --model miphei --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-vit/ --model miphei --dataset hemit --min_area=0
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-vit/ --model miphei --dataset lizard --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/MIPHEI-vit/ --model miphei --dataset pannuke --min_area=0

# Rosie
python run_benchmark.py --checkpoint_dir checkpoints/rosie_orion/ --pred_dir <PRED_DIR_ORION> --model rosie --dataset orion --min_area=10 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/rosie_orion/ --pred_dir <PRED_DIR_PATHOCELL> --model rosie --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/rosie_orion/ --pred_dir <PRED_DIR_HEMIT> --model rosie --dataset hemit --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/rosie_orion/ --pred_dir <PRED_DIR_LIZARD> --model rosie --dataset lizard --min_area=0  --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/rosie_orion/ --pred_dir <PRED_DIR_PANNUKE> --model rosie --dataset pannuke --min_area=0 --num_workers 0

# UpperBound
python run_benchmark.py --checkpoint_dir checkpoints/upperbound --model upperbound --dataset orion --min_area=10
python run_benchmark.py --checkpoint_dir checkpoints/upperbound --model upperbound --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/upperbound --model upperbound --dataset hemit --min_area=0

# Morpho
python run_benchmark.py --checkpoint_dir checkpoints/nuclear_morphometry --model morpho --dataset orion --min_area=10
python run_benchmark.py --checkpoint_dir checkpoints/nuclear_morphometry --model morpho  --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/nuclear_morphometry --model morpho --dataset hemit --min_area=0
python run_benchmark.py --checkpoint_dir checkpoints/nuclear_morphometry --model morpho --dataset lizard --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/nuclear_morphometry --model morpho --dataset pannuke --min_area=0 --num_workers 0

# Diffusion FT
python run_benchmark.py --checkpoint_dir checkpoints/diffusion_ft/ --model diffusion --dataset orion --min_area=10
python run_benchmark.py --checkpoint_dir checkpoints/diffusion_ft/ --model diffusion --dataset pathocell --min_area=0 --num_workers 0
python run_benchmark.py --checkpoint_dir checkpoints/diffusion_ft/ --model diffusion --dataset hemit --min_area=0
python run_benchmark.py --checkpoint_dir checkpoints/diffusion_ft/ --model diffusion --dataset lizard --min_area=0
python run_benchmark.py --checkpoint_dir checkpoints/diffusion_ft/ --model diffusion --dataset pannuke --min_area=0 --num_workers 0
