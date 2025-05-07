"""
Entrypoint script to run trainings.
"""

import argparse
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from src.test import test_model


def main() -> None:
    """
    TO DO
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', help='Checkpoint Path')
    args = parser.parse_args()

    config_path = str(Path(args.checkpoint_dir) / "config.yaml")
    run_name = Path(args.checkpoint_dir).stem
    config = OmegaConf.load(config_path)
    checkpoint_path = str(Path(args.checkpoint_dir) / "model.weights.ckpt")
    test_model(config, checkpoint_path, run_name)




if __name__ == '__main__':
    main()
