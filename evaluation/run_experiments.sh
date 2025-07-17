#!/bin/bash

python eval_orion.py --checkpoint_dir ../checkpoints/MIPHEI-convnext/
python eval_hemit.py --checkpoint_dir ../checkpoints/MIPHEI-convnext/
#python eval_immucan.py --checkpoint_dir ../checkpoints/MIPHEI-convnext/

python eval_orion.py --checkpoint_dir ../checkpoints/UNETR-hoptimus/
python eval_hemit.py --checkpoint_dir ../checkpoints/UNETR-hoptimus/
#python eval_immucan.py --checkpoint_dir ../../checkpoints/UNETR-hoptimus/

python eval_orion.py --checkpoint_dir ../checkpoints/MIPHEI_HEMIT/
python eval_hemit.py --checkpoint_dir ../checkpoints/MIPHEI_HEMIT/
#python eval_immucan.py --checkpoint_dir ../checkpoints/MIPHEI_HEMIT/

python eval_orion.py --checkpoint_dir ../checkpoints/MIPHEI-vit/
python eval_hemit.py --checkpoint_dir ../checkpoints/MIPHEI-vit/
#python eval_immucan.py --checkpoint_dir ../checkpoints/MIPHEI-vit/

python eval_hemit_hemit_pipeline.py --checkpoint_path ../checkpoints/hemit_v1/hemit_v1.pth --trained_hemit

python eval_orion_hemit_pipeline.py --checkpoint_path ../checkpoints/HEMIT-ORION_original/latest_net_G.pth
python eval_hemit_hemit_pipeline.py --checkpoint_path ../checkpoints/HEMIT-ORION_original/latest_net_G.pth
#python eval_immucan_hemit_pipeline.py --checkpoint_path ../checkpoints/HEMIT-ORION_original/latest_net_G.pth
