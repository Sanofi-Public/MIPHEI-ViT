"""
Entrypoint script to run inference.

Predicted images will be saved as TIFF files in checkpoint directory.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.inference import inference_model


def main() -> None:
    """
    Run inference using a specified model checkpoint and dataset configuration.

    This function parses command-line arguments to obtain the checkpoint directory,
    optional dataset-specific configuration, and batch size. It loads the checkpoint and
    dataset-specific configurations, updates configuration parameters as needed,
    and runs inference using the specified model checkpoint.
    Command-line Arguments:
        --checkpoint_dir (str): Path to the directory containing the model checkpoint and
            config.yaml.
        --dataset_config_path (str, optional): Path to an optional dataset-specific configuration
            file. Default will be the dataset from the checkpoint config.
        --batch_size (int, optional): Batch size to use during inference.

    Raises:
        FileNotFoundError: If the provided dataset configuration file does not exist.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', help='Checkpoint Path')
    parser.add_argument('--dataset_config_path', default=None,
                        help='Optional dataset-specific config file (in config/data/).')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Batch size used during inference')
    args = parser.parse_args()

    config_path = str(Path(args.checkpoint_dir) / "config.yaml")
    run_name = Path(args.checkpoint_dir).stem
    config = OmegaConf.load(config_path)

    # Load dataset-specific config if provided
    if args.dataset_config_path:
        if Path(args.dataset_config_path).exists():
            dataset_config = OmegaConf.load(args.dataset_config_path)
            for key in ["slide_dataframe_path", "train_dataframe_path", "val_dataframe_path",
                        "test_dataframe_path"]:
                if key in dataset_config.data:
                    config.data[key] = dataset_config.data[key]
        else:
            raise FileNotFoundError(f"Dataset config {args.dataset_config_path} not found.")

    if args.batch_size:
        config.train["batch_size"] = args.batch_size

    dataset_name = Path(args.dataset_config_path).stem
    args.checkpoint_dir = args.checkpoint_dir
    out_dir_name = f"inference_{dataset_name}_{run_name}"
    out_dir = str(Path(args.checkpoint_dir) / out_dir_name)
    inference_model(config, args.checkpoint_dir, out_dir)


if __name__ == '__main__':
    main()
