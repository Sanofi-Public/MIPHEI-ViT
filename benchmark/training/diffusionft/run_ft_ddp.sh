#!/bin/bash

accelerate launch --num_processes=4 train_ft.py --lambda_pixel_mix_loss 1 --lr_total_iter_length 30000 --dataloader_num_workers 10 --pretrained_model_name_or_path stabilityai/stable-diffusion-2 --output_dir ./runs_ft --seed 42 --train_batch_size 1 --max_train_steps 30000 --gradient_accumulation_steps 8 --checkpointing_steps 1000 --pretrained_unet_checkpoint ./runs --enable_xformers_memory_efficient_attention
