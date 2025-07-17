"""Entrypoint script to run tests."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.test import test_model


def main() -> None:
    """
    Parse command-line argument, load configuration and model checkpoint, and run the model test.

    This function expects a '--checkpoint_dir' argument specifying the directory containing
    the model checkpoint and configuration file. It constructs the paths to the configuration
    YAML file and the model checkpoint file, loads the configuration, and invokes the
    `test_model` function with the loaded configuration, checkpoint path, and run name.
    Raises:
        FileNotFoundError: If the configuration file or checkpoint file does not exist.
        Exception: For errors raised during configuration loading or model testing.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', help='Checkpoint Path')
    args = parser.parse_args()

    config_path = str(Path(args.checkpoint_dir) / "config.yaml")
    run_name = Path(args.checkpoint_dir).stem
    config = OmegaConf.load(config_path)
    checkpoint_path = str(Path(args.checkpoint_dir) / "model.weights.ckpt")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}, make sure you have "
                                "a full checkpoint of Lightning module")
    test_model(config, checkpoint_path, run_name)


if __name__ == '__main__':
    main()
