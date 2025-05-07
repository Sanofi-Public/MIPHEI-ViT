import os
import cv2
import pyvips
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import wandb
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from .utils import save_torch_weights, wandb_log_artifact


class DebugImageLogger(Callback):
    def __init__(self, save_dir, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
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


    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
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
    def predict(self, pl_module, batch):
        log = dict()
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

    def log_img(self, pl_module, batch, batch_idx, split="train"):
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

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (batch_idx > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="test")


class CustomModelCheckpoint(ModelCheckpoint):
    """
    Custom model checkpoint class that extends the base ModelCheckpoint class.
    This class overrides the _save_checkpoint method to save the model state_dict with .pth format.

    Methods:
        _save_checkpoint: Overrides the base class method to save the model state_dict with
        .pth format.
    """

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str):
        """
        Overrides the base class method to save the model state_dict with .pth format.
        """
        # Call the original _save_checkpoint function
        super()._save_checkpoint(trainer, filepath)

        # Save the model state_dict with .pth format
        torchpath = filepath.replace(".ckpt", ".pth")
        save_torch_weights(trainer.model, torchpath)

    def on_train_end(self, trainer, pl_module):
        logger = trainer.logger
        wandb_log_artifact(logger, "model", "model", self.best_model_path)

    def on_exception(self, trainer, pl_module, exception):
        if self.best_model_path is not None:
            logger = trainer.logger
            wandb_log_artifact(logger, "model", "model", self.best_model_path)


class SlideAugentationCallback(Callback):
    def __init__(self, augmentation_slide_dir, prob):
        """
        Args:
            augmentation_slide_dir (std): Path containing the augmented slides.
            prob (float): Probability to apply augmentation.
        """
        self.augmentation_slide_dir = augmentation_slide_dir
        self.prob = prob

    def on_train_start(self, trainer, pl_module):
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

    def on_train_epoch_start(self, trainer, pl_module):
        new_dataframe = self.augment_dataframe()
        trainer.train_dataloader.dataset.df = new_dataframe

    def augment_dataframe(self):
        dataframe = self.dataframe

        def random_augmentation_name(slide_name, prob):
            if np.random.uniform() < prob:
                return slide_name + "_aug"
            return slide_name
        new_dataframe = self.dataframe.copy()
        new_dataframe["in_slide_name"] = dataframe["in_slide_name"].apply(
            lambda x: random_augmentation_name(x, prob=self.prob))
        return new_dataframe


class TileAugentationCallback(Callback):
    def __init__(self, augmentation_tile_dir, prob):
        """
        Args:
            augmentation_tile_dir (std): Path containing the augmented slides.
            prob (float): Probability to apply augmentation.
        """
        self.augmentation_tile_dir = Path(augmentation_tile_dir)
        self.prob = prob

    def on_train_start(self, trainer, pl_module):
        self.dataframe = trainer.train_dataloader.dataset.df.copy()

        new_dataframe = self.augment_dataframe()

        trainer.train_dataloader.dataset.dataframe = new_dataframe

    def on_train_epoch_start(self, trainer, pl_module):
        new_dataframe = self.augment_dataframe()
        trainer.train_dataloader.dataset.df = new_dataframe

    def augment_dataframe(self):

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
    Blabla
    """
    def __init__(self, unormalize_image, num_samples=4):
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

    def setup_callback(self, trainer, pl_module):
        val_dataloader = trainer.val_dataloaders
        val_dataset = val_dataloader.dataset

        # Sample random indices from the validation dataset
        idxs_sampled = np.random.choice(np.arange(len(val_dataset)), self.num_samples, replace=False)
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

    def on_validation_epoch_end(self, trainer, pl_module):
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
        target = torch.clip((self.y + 0.9) / 1.8, 0., 1.)  * 255
        target = torch.permute(target, (0, 2, 3, 1)).to(torch.uint8).numpy()
        target_list = []
        pred_list = []
        #pred_min = preds.min(axis=(0, 1, 2))
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

    def on_train_end(self, trainer, pl_module):
        logger = trainer.logger
        logger.experiment.log({"inference": self.table})


class SavePredictionsCallback(pl.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called at the end of every prediction batch.
        """
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


class SwitchGenDiscTrain(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # After the first epoch, set the boolean attribute to True.
        if epoch == 0 and not trainer.sanity_checking:  # Epochs are zero-indexed in this callback.
            pl_module.gan_train = True
            print(f"Starting to use the discriminator")
