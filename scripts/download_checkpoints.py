"""Script to download model checkpoints and save them into specified directories."""

import argparse
import os
import shutil

import wandb
import subprocess


def check_git_lfs() -> None:
    """
    Check if Git LFS is installed and accessible.

    Raises:
        RuntimeError: If Git LFS is not installed.
        subprocess.CalledProcessError: If Git LFS is installed but returns an error.
    """
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("✅ Git LFS is installed:", result.stdout.decode().strip())
    except subprocess.CalledProcessError:
        print("❌ Git LFS is installed but returned an error.")
        raise
    except FileNotFoundError:
        print("❌ Git LFS is not installed. Please install it first:\n  https://git-lfs.github.com/")
        raise RuntimeError("Git LFS not installed.")


def check_gdown():
    """
    Check if the 'gdown' command-line tool is installed and available.

    Raises:
        RuntimeError: If 'gdown' is not installed.
        subprocess.CalledProcessError: If 'gdown' exists but returns an error.
    """
    try:
        result = subprocess.run(
            ["gdown", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("✅ gdown is installed:", result.stdout.decode().strip())
    except subprocess.CalledProcessError:
        print("❌ gdown command exists but returned an error.")
        raise
    except FileNotFoundError:
        print(
            "❌ gdown is not installed.\n"
            "You can install it with:\n"
            "   pip install gdown"
        )
        raise RuntimeError("gdown not installed.")


if __name__ == "__main__":
    check_git_lfs()
    check_gdown()
    parser = argparse.ArgumentParser(
        description="Download checkpoints into the specified directory")
    parser.add_argument("--output_dir", type=str, default="../checkpoints",
                        help="Directory to save the downloaded checkpoints")
    args = parser.parse_args()

    run = wandb.init()

    models = [
        ("MIPHEI-vit", "guillaume-balezo/MIPHEI-ViT_paper/MIPHEI-vit:v0"),
        ("UNETR-hoptimus", "guillaume-balezo/MIPHEI-ViT_paper/UNETR-hoptimus:v0"),
        ("MIPHEI_HEMIT", "guillaume-balezo/MIPHEI-ViT_paper/MIPHEI_HEMIT:v0"),
        ("HEMIT-ORION_original", "guillaume-balezo/MIPHEI-ViT_paper/HEMIT-ORION_original:v0"),
        ("MIPHEI-convnext", "guillaume-balezo/MIPHEI-ViT_paper/MIPHEI-convnext:v0"),
        ("HEMIT", "guillaume-balezo/MIPHEI-ViT_paper/HEMIT:v0"),
        ("Pix2Pix", "guillaume-balezo/MIPHEI-ViT_paper/Pix2Pix:v0"),
        ("DiffusionFT", "guillaume-balezo/MIPHEI-ViT_paper/DiffusionFT:v0"),
        ("Rosie-ORION", "guillaume-balezo/MIPHEI-ViT_paper/ROSIE-OrionCRC:v0"),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, artifact_path in models:
        dest_dir = os.path.join(args.output_dir, model_name)
        if os.path.exists(dest_dir):
            print(f"{model_name} already exists in {args.output_dir}, skipping download.")
            continue
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        dest_dir = os.path.join(args.output_dir, model_name)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.move(artifact_dir, dest_dir)

    wandb.finish()
    shutil.rmtree("artifacts")  # Clean up the wandb directory
    shutil.rmtree("wandb")  # Clean up the wandb directory

    dest_dir = os.path.join(args.output_dir, "hemit_v1", "hemit_v1.pth")
    if not os.path.exists(dest_dir):
        # Create hemit_v1 folder
        hemit_v1_dir = os.path.join(args.output_dir, "hemit_v1")
        os.makedirs(hemit_v1_dir, exist_ok=True)
        # Download Google Drive file using gdown
        gdown_cmd = [
            "gdown",
            "--id", "1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd",
            "-O", os.path.join(hemit_v1_dir, "hemit_v1.pth")
        ]
        subprocess.run(gdown_cmd, check=True)
    else:
        print(f"hemit_v1 already exists in {args.output_dir}, skipping download.")