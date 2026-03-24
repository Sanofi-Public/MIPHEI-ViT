#!/usr/bin/env python3

import os
import argparse
import urllib.request

RELEASE = "v1.0.0"
BASE_URL = f"https://github.com/Sanofi-Public/MIPHEI-ViT/releases/download/{RELEASE}"

FILES = [
    "model.safetensors",
    "config.yaml",
    "LICENSE",
    "logreg.pth",
    "model.py",
    "requirements.txt",
]


def download(url, filepath):
    try:
        print(f"  ⬇️  Downloading {os.path.basename(filepath)} ... ", end="", flush=True)
        urllib.request.urlretrieve(url, filepath)
        print("✓")
    except Exception as e:
        print("✗ (failed)")
        print(f"    Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download MIPHEI-ViT release assets")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory where files will be downloaded (default: current directory)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"📦 Downloading MIPHEI-ViT release assets ({RELEASE})...")
    print(f"📁 Destination: {os.path.abspath(out_dir)}\n")

    for fname in FILES:
        url = f"{BASE_URL}/{fname}"
        filepath = os.path.join(out_dir, fname)
        download(url, filepath)

    print("\n✔ All downloads completed.")


if __name__ == "__main__":
    main()