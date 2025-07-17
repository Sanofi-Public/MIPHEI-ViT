"""Implementation of Pytorch Lightning callbacks used during training and evaluation."""

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyvips
import torch
import wandb
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid


class DebugImageLogger(Callback):
    """
    DebugImageLogger is a PyTorch Lightning Callback for logging and saving image reconstructions \
    during model training and testing.

    This callback periodically saves input, target, and predicted images to disk, which is useful
    for debugging and visualizing model outputs. It supports configurable logging frequency,
    maximum number of images saved, and options for image rescaling and clamping.
    The callback can be enabled or disabled, and supports both training and testing phases.

    Args:
        save_dir (str): Directory where images will be saved.
        batch_frequency (int): Frequency (in batches) at which images are logged.
        max_images (int): Maximum number of images to log.
        clamp (bool, optional): Whether to clamp image values to [-1, 1] before saving.
            Default is True.
        increase_log_steps (bool, optional): If True, logging frequency increases exponentially at
            early steps. Default is True.
        rescale (bool, optional): Whether to rescale images from [-1, 1] to [0, 1] before saving.
            Default is True.
        disabled (bool, optional): If True, disables logging. Default is False.
        log_on_batch_idx (bool, optional): If True, uses batch index for logging frequency instead
            of global step. Default is False.
        log_first_step (bool, optional): If True, logs images at the first step. Default is False.
        log_images_kwargs (dict, optional): Additional keyword arguments for image logging.
    Methods:
        log_local(save_dir, split, images, global_step, current_epoch, batch_idx):
            Saves a grid of images to the specified directory.
        predict(pl_module, batch):
            Generates model predictions for the given batch and formats them for logging.
        log_img(pl_module, batch, batch_idx, split="train"):
            Handles the process of logging images for a given batch.
        check_frequency(check_idx):
            Determines whether the current step or batch index meets the logging frequency criteria.
        on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            Called at the end of each training batch to potentially log images.
        on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            Called at the end of each test batch to potentially log images.
    Usage:
        Add this callback to the PyTorch Lightning Trainer to automatically save and log images
            during training and testing for debugging and visualization purposes.
    """

    def __init__(self, save_dir: str, batch_frequency: int, max_images: int, clamp: bool = True,
                 increase_log_steps: bool = True, rescale: bool = True, disabled: bool = False,
                 log_on_batch_idx: bool = False, log_first_step: bool = False,
                 log_images_kwargs: dict = None):
        super().__init__()
        self.save_dir = save_dir
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def log_local(self, save_dir: str, split: str, images: torch.Tensor,
                  global_step: int, current_epoch: int, batch_idx: int) -> None:
        """Log and save image grids locally for a given split."""
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad
    def predict(self, pl_module: pl.LightningModule, batch: dict) -> dict:
        """Generate predictions and log predictions and targets for a given batch."""
        log = {}
        x = batch["image"]
        x = x.to(pl_module.device)
        y = batch["target"]

        if pl_module.foreground_head:
            y_fake, _ = pl_module.generator(x.to(pl_module.device))
            y_fake = y_fake.cpu()
        else:
            y_fake = pl_module.generator(x.to(pl_module.device)).cpu()
        batch_size, n_c, h, w = y_fake.shape
        log["reconstructions"] = y_fake.reshape((batch_size, 1, -1, w)) / 0.9
        log["targets"] = y.reshape((batch_size, 1, -1, w)) / 0.9
        return log

    def log_img(self, pl_module: pl.LightningModule, batch: dict, batch_idx: int,
                split: str = "train") -> None:
        """Log images at intervals during training or evaluation."""
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                batch_idx > 5 and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = self.predict(pl_module, batch)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(self.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx: int) -> bool:
        """Determine whether to do log operation at the given batch step."""
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs,
                           batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Run callback at end of training batch if enabled."""
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs,
                          batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Run callback at end of test batch if enabled."""
        if not self.disabled and (batch_idx > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="test")


class SlideAugmentationCallback(Callback):
    """
    Callback to augment training WSIs with a specified probability during model training.

    This callback is for use with `Img2ImgSlideDataset` from our SlideVips package when training
    with precomputed augmented WSIs (e.g., heavy augmentation like CycleGAN). At training start, it
    updates slide path mappings to include both original and augmented slides. At each epoch, it
    probabilistically replaces original slides with their augmented versions based on a set
    probability.
    Args:
        augmentation_slide_dir (str): Directory containing the augmented WSIs.
        prob (float): Probability of applying augmentation to a slide.
    Attributes:
        augmentation_slide_dir (str): Directory containing the augmented slides.
        prob (float): Probability of applying augmentation to a slide.
        augmentation_slide_dir (str): Path containing the augmented slides.
        prob (float): Probability to apply augmentation to each slide.
    """

    def __init__(self, augmentation_slide_dir: str, prob: float):
        self.augmentation_slide_dir = augmentation_slide_dir
        self.prob = prob

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Augments dataset slide path mappings with augmented slides at the start of training."""
        self.dataframe = trainer.train_dataloader.dataset.df.copy()

        inslide_name2path = trainer.train_dataloader.dataset.inslide_name2path.copy()
        aug_inslide_name2path = inslide_name2path.copy()
        targslide_name2path = trainer.train_dataloader.dataset.targslide_name2path.copy()
        aug_targslide_name2path = targslide_name2path.copy()
        for slide_name, slide_path in inslide_name2path.items():
            aug_slide_name = slide_name + "_aug"
            aug_slide_path = str(Path(self.augmentation_slide_dir) / Path(slide_path).name)
            aug_inslide_name2path[aug_slide_name] = aug_slide_path
            aug_targslide_name2path[aug_slide_name] = targslide_name2path[slide_name]
        trainer.train_dataloader.dataset.inslide_name2path = aug_inslide_name2path
        trainer.train_dataloader.dataset.targslide_name2path = aug_targslide_name2path

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Augments the training dataset's dataframe at the start of each training epoch."""
        new_dataframe = self.augment_dataframe()
        trainer.train_dataloader.dataset.df = new_dataframe

    def augment_dataframe(self) -> pd.DataFrame:
        """Randomly replace 'in_slide_name' with its augmented version based on probability."""
        dataframe = self.dataframe

        def random_augmentation_name(slide_name, prob):
            if np.random.uniform() < prob:
                return slide_name + "_aug"
            return slide_name
        new_dataframe = self.dataframe.copy()
        new_dataframe["in_slide_name"] = dataframe["in_slide_name"].apply(
            lambda x: random_augmentation_name(x, prob=self.prob))
        return new_dataframe


class TileAugmentationCallback(Callback):
    """
    Callback to apply precomputed tile augmentations during training.

    This callback modifies the image paths in the training dataframe dataset to point to
    precomputed augmented images stored in a specified directory, based on a given probability.
    This is particularly useful for heavy augmentations (e.g., those generated by CycleGAN) that
    are expensive to compute on the fly.
    Attributes:
        augmentation_tile_dir (Path): Directory containing the augmented image tiles.
        prob (float): Probability of replacing an image path with its augmented counterpart.
        augmentation_tile_dir (str or Path): Path to the directory containing augmented tiles.
        prob (float): Probability to apply augmentation to each image.
    Methods:
        on_train_start(trainer, pl_module): Called at the start of training to initialize the
            dataframe with possible augmentations.
        on_train_epoch_start(trainer, pl_module): Called at the start of each training epoch to
            potentially re-apply augmentations.
        augment_dataframe(): Returns a new dataframe with image paths replaced by augmented
            versions based on the specified probability.
    """

    def __init__(self, augmentation_tile_dir: str, prob: float):
        self.augmentation_tile_dir = Path(augmentation_tile_dir)
        self.prob = prob

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Augment and update the training dataset's dataframe at training start."""
        self.dataframe = trainer.train_dataloader.dataset.df.copy()
        new_dataframe = self.augment_dataframe()
        trainer.train_dataloader.dataset.dataframe = new_dataframe

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Augments the training dataset's dataframe at the start of each training epoch."""
        new_dataframe = self.augment_dataframe()
        trainer.train_dataloader.dataset.df = new_dataframe

    def augment_dataframe(self) -> pd.DataFrame:
        """Return a dataframe with image paths randomly replaced by augmented versions."""

        def random_augmentation_name(image_path, prob):
            if np.random.uniform() < prob:
                return str(self.augmentation_tile_dir / Path(image_path).name)
            return image_path
        new_dataframe = self.dataframe.copy()
        new_dataframe["image_path"] = new_dataframe["image_path"].apply(
            lambda x: random_augmentation_name(x, prob=self.prob))
        return new_dataframe


class WandbVisCallback(Callback):
    """
    Callback for visualizing input H&E, target mIF, and predicted mIF images in Weights & Biases.

    This callback samples a fixed subset of images from the validation dataset at the start of
    training, and uses the same samples for visualization at the end of each validation epoch. For
    each sample, the input image, target channels, and predicted channels (stacked vertically) are
    displayed. Structural Similarity Index (SSIM) is also computed and logged.
    Attributes:
        unormalize_image (Callable): Function to unnormalize input images for visualization.
        num_samples (int): Number of samples to visualize from the validation set.
        x (torch.Tensor): Sampled input images (fixed for all epochs).
        y (torch.Tensor): Sampled target images (fixed for all epochs).
        img_shape (tuple): Shape of the images.
        ssim_metric (StructuralSimilarityIndexMeasure): SSIM metric instance for evaluation.
        table (wandb.Table): Wandb table for logging results.
        device (torch.device): Device on which the model is running.
        foreground_head (bool): Indicates if the model uses a foreground head.
    Methods:
        setup_callback(trainer, pl_module):
            Prepares the callback by sampling images from the validation dataset and initializing
                logging structures.
        on_validation_epoch_end(trainer, pl_module):
            Runs at the end of each validation epoch to generate predictions, compute SSIM, and log
                images and metrics to Wandb.
        on_train_end(trainer, pl_module):
            Logs the accumulated Wandb table at the end of training.
    """

    def __init__(self, unormalize_image: Callable, num_samples: int = 4):
        self.num_samples = num_samples
        self.unormalize_image = unormalize_image
        self.x = None
        self.y = None
        self.img_shape = None
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=(-0.9, 0.9))
        self.table = None
        self.device = None
        self.foreground_head = None
        self.num_samples = num_samples

    def setup_callback(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Set up callback: sample validation data, initialization, and preparing Wandb logging."""
        val_dataloader = trainer.val_dataloaders
        val_dataset = val_dataloader.dataset

        # Sample random indices from the validation dataset
        idxs_sampled = np.random.choice(np.arange(len(val_dataset)), self.num_samples,
                                        replace=False)
        x, y = [], []
        for idx in idxs_sampled:
            data = val_dataset[idx]
            x.append(data["image"])
            y.append(data["target"])

        # Store the sampled images and targets as tensors
        self.x = torch.stack(x, dim=0)
        self.y = torch.stack(y, dim=0)
        val_dataset.reset()

        # Set up other necessary attributes
        nc_out = self.y.shape[1]
        self.img_shape = self.y.shape[2:]

        # Initialize the Wandb table with proper columns
        table_columns = ["epoch", "ssim", "image"]
        for idx_marker in range(nc_out):
            table_columns.append(f"marker_{idx_marker}")
        self.table = wandb.Table(columns=table_columns)

        logger = trainer.logger
        assert isinstance(logger, WandbLogger)
        self.device = pl_module.device
        self.foreground_head = pl_module.foreground_head

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Inference, post-process predictions, and log images."""
        if self.x is None:
            self.setup_callback(trainer, pl_module)
        epoch = trainer.current_epoch
        logger = trainer.logger
        with torch.no_grad():
            if self.foreground_head:
                preds, _ = pl_module.generator(self.x.to(self.device))
            else:
                preds = pl_module.generator(self.x.to(self.device))
            preds = preds.cpu().float()
            ssim = self.ssim_metric(preds, self.y)
            preds = torch.clip((preds + 0.9) / 1.8, 0., 1.) * 255
            preds = torch.permute(preds, (0, 2, 3, 1)).to(torch.uint8).numpy()
        image = torch.permute(self.x, (0, 2, 3, 1)).numpy()
        image = np.uint8(self.unormalize_image(image))
        target = torch.clip((self.y + 0.9) / 1.8, 0., 1.) * 255
        target = torch.permute(target, (0, 2, 3, 1)).to(torch.uint8).numpy()
        target_list = []
        pred_list = []
        #  pred_min = preds.min(axis=(0, 1, 2))
        for idx_channel in range(preds.shape[-1]):
            pred_curr = np.repeat(preds[..., idx_channel, np.newaxis], 3, axis=-1)
            pred_list.append(pred_curr)
            target_curr = np.repeat(target[..., idx_channel, np.newaxis], 3, axis=-1)
            target_list.append(target_curr)

        for idx in range(len(preds)):
            data_table = [epoch, ssim, wandb.Image(image[idx])]
            for idx_marker in range(len(pred_list)):
                data_table.append(wandb.Image(pred_list[idx_marker][idx]))
            self.table.add_data(*data_table)

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        concatenated_images = np.concatenate([image] + target_list + pred_list, axis=1)
        concatenated_images = torch.permute(torch.from_numpy(
            concatenated_images), (0, 3, 1, 2))
        grid = make_grid(concatenated_images,
                         nrow=len(concatenated_images), value_range=(0, 255))
        grid = grid.permute((1, 2, 0)).numpy()
        """scale_grid = 224 / image.shape[1]
        if scale_grid < 1:
            grid = cv2.resize(grid, dsize=None, fx=scale_grid,
                            fy=scale_grid, interpolation=cv2.INTER_LINEAR)"""
        if not trainer.sanity_checking:
            logger.experiment.log({f"image_{epoch}": [
                wandb.Image(grid, caption="Input - Preds - Target", file_type="jpg")]})

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the inference table to WandB at the end of training."""
        logger = trainer.logger
        logger.experiment.log({"inference": self.table})


class SavePredictionsCallback(pl.Callback):
    """
    Callback to save model predictions as TIFF files during prediction.

    This callback is designed to be used with PyTorch Lightning's prediction loop.
    After each prediction batch, it saves the predicted images to the specified output directory,
    normalizing them to uint8 and using the corresponding tile names from the batch.
    Args:
        output_dir (str): Directory where the predicted TIFF files will be saved.
    Attributes:
        output_dir (Path): Path object representing the output directory.
    Methods:
        on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            Called at the end of every prediction batch to save predictions as TIFF files.
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def on_predict_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs,
                             batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Process and save each prediction batch as TIFF images at the end of prediction."""
        # Extract images & tile names
        prediction_batch = outputs  # Should be the generator output
        tile_names_batch = batch["tile_name"]

        # Normalize to uint8
        prediction_batch = ((prediction_batch + 0.9) / 1.8).clamp(0, 1)  # Ensure values in [0,1]
        prediction_batch = (prediction_batch * 255).to(torch.uint8).cpu().numpy()

        # Save each tile as a TIFF file
        for prediction, tile_name in zip(prediction_batch, tile_names_batch):
            out_path = self.output_dir / f"{tile_name}.tiff"
            pyvips.Image.new_from_array(prediction).write_to_file(str(out_path))


class SwitchGenDiscTrain(Callback):  # Not used in the current codebase
    """
    Callback to switch on GAN discriminator training after the first epoch.

    This callback sets the `gan_train` attribute of the `pl_module` to `True`
    at the end of the first training epoch, enabling the training of the discriminator
    in a GAN setup. It ensures that the discriminator is only used after the initial
    epoch, which can be useful for stabilizing GAN training.
    Args:
        Callback: Inherits from the base PyTorch Lightning Callback class.
    Methods:
        on_train_epoch_end(trainer, pl_module):
            Sets `pl_module.gan_train` to `True` after the first epoch.
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Enable GAN training after the first epoch."""
        epoch = trainer.current_epoch
        # After the first epoch, set the boolean attribute to True.
        if epoch == 0 and not trainer.sanity_checking:  # Epochs are zero-indexed in this callback.
            pl_module.gan_train = True
            print("Starting to use the discriminator")
