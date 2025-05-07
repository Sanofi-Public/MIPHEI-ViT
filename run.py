"""
Entrypoint script to run trainings.
"""

from datetime import datetime
import os
from pathlib import Path
import shutil
import yaml
import hydra
from omegaconf import DictConfig
import wandb

from src.train import train_patchgan

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for training the model.

    This function parses command-line arguments, loads the configuration file,
    and calls the appropriate training function based on the specified mode.

    Args:
        None

    Returns:
        None
    """
    # Parse command-line arguments

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("logs").mkdir(exist_ok=True)
    logdir = Path("logs") / "patchgan_{}_{}".format(
        "_".join(map(str, cfg.data.targ_channel_names)), timestamp)
    logdir.mkdir()
    #  TODO: check validity of config
    with open(str(logdir / "status.txt"), "w") as f:
        f.write("not finished")
    #shutil.copy(args.config_path, str(logdir / "config.yaml"))
    write_github_logs(logdir)
    train_patchgan(cfg, str(logdir))

    with open(str(logdir / "status.txt"), "w") as f:
            f.write("finished")
    wandb.finish()


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration.

    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_github_logs(logdir):
    github_file = str(logdir / "github_log.txt")
    os.system(f"git rev-parse --short HEAD >>{github_file}")
    os.system(f"git diff HEAD >>{github_file}")


if __name__ == '__main__':
    main()
