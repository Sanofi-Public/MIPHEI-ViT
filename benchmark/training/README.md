# ⚙️ Training the Models

Below are the scripts used to train all benchmarked models.

---

## **MIPHEI-ViT**

Training is done directly from the main repo:

```bash
python run.py +default_configs=miphei-vit
```

---

## **DiffusionFT**

Located in `diffusionft/`.

**Step 1 — Diffusion pretraining**

```bash
bash diffusionft/run_train_ddp.sh
```

**Step 2 — Fine-tuning**

```bash
bash diffusionft/run_ft_ddp.sh
```

Dependencies: MIPHEI requirements + `accelerate`.
Code adapted from https://github.com/VisualComputingInstitute/diffusion-e2e-ft

---

## **ROSIE**

Located in `rosie/`.

```bash
python rosie/train_orion.py
```

Dependencies: MIPHEI requirements.
Code adapted from https://gitlab.com/enable-medicine-public/rosie

---

## **HEMIT (Dual-Branch Pix2Pix)**

Based on the original implementation:

https://github.com/BianChang/Pix2pix_DualBranch

You must first set up the environment and dependencies following the instructions of the original repo.

Adapt scripts according to `hemit/diff.txt`, then run:

```bash
bash hemit/train_hemit_orion.sh.
```

To evaluate all epochs and select the best checkpoint on validation:

```bash
bash hemit/test_all_hemit_orion.sh
```

This script creates an `eval.txt` file inside the checkpoint directory containing the validation L1 loss for each epoch, allowing you to choose the best model.

You may need to adjust `dataroot` in the script

---

## **Pix2Pix (Baseline)**

Based on the official repo:

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

You must first set up the environment and dependencies following the instructions of the original repo.

Training script:

```bash
bash pix2pix/train_pix2pix_orion.sh
```

To evaluate all epochs and select the best checkpoint on validation:

```bash
bash pix2pix/test_all_pix2pix_orion.sh
```

This also generates an `eval.txt` in the checkpoint directory with validation L1 losses to identify the best epoch.

Again, update `dataroot` as needed.
