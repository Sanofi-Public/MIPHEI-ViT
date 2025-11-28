"""
Model classes for image to image translation.

Used here to define Lightning Module to predict mIF from H&E images.
Contain also the discriminator module for GAN training.
The model module allows the implementation of a Pix2Pix-like architecture.
"""

import functools
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .loss import FocalLoss
from .utils import MeanCellExtrator, pix2pix_lr_scheduler  # get_vit_lr_decay_rate
# from .generators.unetr import ViTPyramidEncoder


class ModelModule(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating image-to-image translation models, \
    optionally with GAN and cell-level metrics support.

    This module encapsulates the training, validation, and testing logic for a generator
    (possibly with a foreground segmentation head) and an optional discriminator. It supports
    pixel-level metrics (PSNR, SSIM), adversarial training (Pix2Pix), perceptual loss, and
    cell-level metrics for biological image analysis.
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        lr_g (float): Learning rate for the generator.
        lr_d (float): Learning rate for the discriminator.
        loss_reconstruct (callable): Reconstruction loss function.
        cell_metrics (optional): Metrics for cell-level evaluation.
        cell_loss (optional): Loss function for cell-level evaluation.
        gan_train (bool): Whether to enable GAN training.
    Attributes:
        generator (nn.Module): The generator network.
        discriminator (nn.Module or None): The discriminator network (if GAN training is enabled).
        lr_g (float): Learning rate for the generator.
        lr_d (float): Learning rate for the discriminator.
        loss_reconstruct (callable): Reconstruction loss function.
        cell_metrics (optional): Metrics for cell-level evaluation.
        use_cell_metrics (bool): Whether to use cell-level metrics.
        cell_loss (optional): Loss function for cell-level evaluation.
        mean_cell_extractor (optional): Extractor for mean cell values.
        gan_train (bool): Whether to enable GAN training.
        perceptual_loss_fn (optional): Perceptual loss function.
        train_pix_metrics (MetricCollection): Pixel-level metrics for training.
        train_disc_metrics (MetricCollection): Discriminator metrics for training.
        val_pix_metrics (MetricCollection): Pixel-level metrics for validation.
        val_disc_metrics (MetricCollection): Discriminator metrics for validation.
        test_pix_metrics (MetricCollection): Pixel-level metrics for testing.
        test_disc_metrics (MetricCollection): Discriminator metrics for testing.
        logreg_layer (nn.Linear): Logistic regression layer for cell-level metrics.
        is_lsgan (bool): Whether to use least squares GAN loss.
    Methods:
        forward(inputs): Forward pass through the generator.
        predict_step(batch, batch_idx): Inference step for prediction.
        adversarial_loss(target, input): Computes adversarial loss (BCE or MSE).
        training_step(batch, batch_idx): Training logic for generator and discriminator.
        on_train_epoch_end(): Computes and logs training metrics at epoch end.
        evaluation_step(batch, batch_idx, prefix): Shared logic for validation and test steps.
        validation_step(batch, batch_idx): Validation step for Lightning.
        test_step(batch, batch_idx): Test step for Lightning.
        epoch_end_cell_metrics(prefix, logreg_layer=None, return_dataframe=False): Computes and
            logs cell metrics.
        on_validation_epoch_end(): Computes and logs validation metrics at epoch end.
        on_test_epoch_end(): Computes and logs test metrics at epoch end.
        configure_optimizers(): Configures optimizers and schedulers for generator and
            discriminator.
        _log_train_metric(metric_name, metric_value): Helper to log training metrics.
        _log_val_metric(metric_name, metric_value): Helper to log validation metrics.
    Raises:
        ValueError: If NaN values are detected in the generator output during training.
    """

    def __init__(self, generator: nn.Module, discriminator: nn.Module, lr_g: float, lr_d: float,
                 loss_reconstruct, cell_metrics: bool = None, cell_loss=None,
                 gan_train: bool = False):
        super().__init__()
        self.generator = generator
        self.foreground_head = hasattr(generator, "foreground_head")
        if self.foreground_head:
            self.foreground_loss = FocalLoss(alpha=0.75, gamma=2.)
        self.discriminator = discriminator if gan_train else None
        self.perceptual_loss_fn = None
        self.automatic_optimization = False  # Handle optimizers manually
        self.gan_train = gan_train

        # Metrics
        self.train_pix_metrics = MetricCollection(
            {
                "psnr_metric": PeakSignalNoiseRatio(data_range=(-0.9, 0.9)),
                "ssim_metric": StructuralSimilarityIndexMeasure(data_range=(-0.9, 0.9)),
            },
            prefix="train_",
        )
        self.train_disc_metrics = MetricCollection(
            {
                "precision": BinaryPrecision(),
                "recall_metric": BinaryRecall(),
            },
            prefix="train_",
        )
        self.val_pix_metrics = self.train_pix_metrics.clone(prefix="val_")
        self.val_disc_metrics = self.train_disc_metrics.clone(prefix="val_")
        self.test_pix_metrics = self.train_pix_metrics.clone(prefix="test_")
        self.test_disc_metrics = self.train_disc_metrics.clone(prefix="test_")

        self.use_cell_metrics = True if cell_metrics is not None else False
        self.cell_metrics = cell_metrics
        self.cell_loss = cell_loss
        if self.use_cell_metrics:
            self.logreg_layer = nn.Linear(
                len(self.cell_metrics.pred_marker_names), len(self.cell_metrics.target_names))
            if self.cell_loss is not None:
                self.mean_cell_extractor = MeanCellExtrator()

        # End metrics
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.loss_reconstruct = loss_reconstruct

        """
        # DEPRECATED
        if hasattr(self.generator, "encoder"):  # DEPRECATED
            self.vit_lr_decay = isinstance(self.generator.encoder, ViTPyramidEncoder) and \
                all(param.requires_grad for param in self.generator.encoder.model.parameters())
        else:
            self.vit_lr_decay = False
        """
        self.is_lsgan = False

    def forward(self, inputs: torch.Tensor):
        """Pass inputs through the generator network."""
        return self.generator(inputs)

    def predict_step(self, batch: dict, batch_idx: int):
        """Inference step for prediction."""
        return self.generator(batch["image"])

    def adversarial_loss(self, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss using either LSGAN (MSE) or standard GAN (BCE)."""
        if self.is_lsgan:
            return F.mse_loss(target=target, input=input)
        else:
            return F.binary_cross_entropy_with_logits(target=target, input=input)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        """Perform a single training step for the Lightning module."""
        x, y = batch["image"], batch["target"]

        if self.gan_train:
            g_optimizer, d_optimizer = self.optimizers()
            g_scheduler, d_scheduler = self.lr_schedulers()
        else:
            g_optimizer = self.optimizers()
            g_scheduler = self.lr_schedulers()
        self.toggle_optimizer(g_optimizer)
        if self.foreground_head:
            fake_images, foreground_preds = self.generator(x)
        else:
            fake_images = self.generator(x)

        with torch.no_grad():
            if torch.isnan(fake_images).any():
                torch.save(self.generator.state_dict(), "weights_nan.ckpt")
                raise ValueError("Nan found")
        # Generator step
        if self.gan_train:
            disc_output_fake = self.discriminator(x, fake_images)
            misleading_labels = torch.zeros(disc_output_fake.shape).type_as(x)
            gen_adv_loss = self.adversarial_loss(target=misleading_labels, input=disc_output_fake)
        else:
            gen_adv_loss = 0.
        gen_loss_sim = self.loss_reconstruct(y_true=y, y_pred=fake_images)
        gen_loss = gen_loss_sim + gen_adv_loss
        if self.use_cell_metrics and (self.cell_loss is not None):
            nuclei_masks = batch["nuclei"]
            pred_cell_means, target_cell_means, _ = self.mean_cell_extractor(
                pred=fake_images, target=y, nuclei=nuclei_masks)
            loss_cell_value = self.cell_loss(pred_cell_means, target_cell_means)
            gen_loss += loss_cell_value

        if self.perceptual_loss_fn:
            loss_p = self.perceptual_loss_fn.forward(
                fake_images.contiguous(), y.contiguous())
            gen_loss = gen_loss + 0.1 * loss_p

        if self.foreground_head:
            target_foreground = (y > -0.9).type_as(y)
            """target_foreground = F.max_pool2d(
                target_foreground, 5, stride=1, padding=2)"""
            gen_foreground_loss = self.foreground_loss(
                target=target_foreground, input=foreground_preds)
            gen_loss = gen_loss + gen_foreground_loss
        g_optimizer.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(g_optimizer, gradient_clip_val=1., gradient_clip_algorithm="norm")
        g_optimizer.step()
        g_scheduler.step()
        self.untoggle_optimizer(g_optimizer)
        with torch.no_grad():
            y_clip = y.contiguous()
            fake_images_clip = fake_images.contiguous().clip(-0.9, 0.9)
            self.train_pix_metrics.update(fake_images_clip, y_clip)
            # if self.foreground_head:
            #    foreground_pred_class = F.sigmoid(foreground_preds) > 0.5
            #    dice_value = self.dice_metric(foreground_pred_class, y_clip > -0.9)

        # Discriminator step
        if self.gan_train:
            self.toggle_optimizer(d_optimizer)

            disc_output_fake = self.discriminator(
                x, fake_images.detach())
            disc_output_real = self.discriminator(
                x, y)

            fake_labels = torch.ones(disc_output_fake.shape).type_as(x)
            fake_labels = fake_labels + 0.05 * torch.rand(fake_labels.shape).type_as(x)
            fake_labels = torch.clip(fake_labels, 0., 1.)
            disc_fake_adv_loss = self.adversarial_loss(target=fake_labels, input=disc_output_fake)
            real_labels = torch.zeros(disc_output_real.shape).type_as(x)
            real_labels = real_labels + 0.05 * torch.rand(real_labels.shape).type_as(x)
            real_labels = torch.clip(real_labels, 0., 1.)
            disc_real_adv_loss = self.adversarial_loss(target=real_labels, input=disc_output_real)
            disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
            d_optimizer.zero_grad()
            self.manual_backward(disc_adv_loss)
            self.clip_gradients(d_optimizer, gradient_clip_val=1., gradient_clip_algorithm="norm")
            d_optimizer.step()
            d_scheduler.step()
            self.untoggle_optimizer(d_optimizer)

            if not self.is_lsgan:
                with torch.no_grad():
                    disc_output = torch.concat([disc_output_fake, disc_output_real], axis=0)
                    disc_pred = F.sigmoid(disc_output) > 0.5
                    real_labels = torch.concat(
                        [
                            torch.ones(disc_output_fake.shape).type_as(x),
                            torch.zeros(disc_output_real.shape).type_as(x),
                        ],
                        axis=0,
                    )
                    self.train_disc_metrics.update(disc_pred, real_labels)

        # Log metrics (includes the metric that tracks the loss)
        self.log('lr', g_scheduler.get_last_lr()[0], on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)
        self.log("gen_loss_sim_step", gen_loss_sim, on_step=True, on_epoch=False,
                 prog_bar=True, logger=False)
        self.log("train_gen_loss", gen_loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_gen_loss_sim", gen_loss_sim, on_step=False, on_epoch=True, logger=True)

        if self.gan_train:
            self.log("train_gen_adv_loss", gen_adv_loss, on_step=False, on_epoch=True, logger=True)
            self.log("train_disc_adv_loss_step", disc_adv_loss, on_step=True, on_epoch=False,
                     prog_bar=True, logger=False)
            self.log("train_disc_adv_loss", disc_adv_loss, on_step=False, on_epoch=True,
                     logger=True)

        if self.use_cell_metrics and (self.cell_loss is not None):
            self.log("train_loss_cell", loss_cell_value, on_step=False, on_epoch=True,
                     logger=True, prog_bar=True)
        if self.foreground_head:
            self.log("train_gen_foreground_loss", gen_foreground_loss, on_step=False,
                     on_epoch=True, logger=True)
        if self.perceptual_loss_fn:
            self.log("train_loss_p", loss_p, on_step=False, on_epoch=True, logger=True)

    def on_train_epoch_end(self):
        """Handle end-of-epoch training logic by logging and resetting metrics."""
        # self.log_dict(out_pix_metrics, on_step=False, on_epoch=True, logger=True)
        self.log_dict(self.train_pix_metrics.compute())
        self.log_dict(self.train_disc_metrics.compute())
        self.train_pix_metrics.reset()
        self.train_disc_metrics.reset()
        super().on_train_epoch_end()

    def evaluation_step(self, batch: dict, batch_idx: int, prefix: str) -> None:
        """Perform a single evaluation step for the Lightning module."""
        x, y = batch["image"], batch["target"]

        if self.foreground_head:
            fake_images, foreground_preds = self.generator(x)
        else:
            fake_images = self.generator(x)

        # Generator step
        if self.gan_train:
            disc_output_fake = self.discriminator(x, fake_images)
            misleading_labels = torch.zeros(disc_output_fake.shape).type_as(x)
            gen_adv_loss = self.adversarial_loss(target=misleading_labels, input=disc_output_fake)
        else:
            gen_adv_loss = 0.
        gen_loss_sim = self.loss_reconstruct(y_true=y, y_pred=fake_images)
        gen_loss = gen_loss_sim + gen_adv_loss

        if self.use_cell_metrics:
            nuclei_masks = batch["nuclei"]
            slide_names = batch["slide_name"]
            if self.cell_loss is not None:
                pred_cell_means, target_cell_means, _ = self.mean_cell_extractor(
                    pred=fake_images, target=y, nuclei=nuclei_masks)
                loss_cell_value = self.cell_loss(pred_cell_means, target_cell_means)
                gen_loss += loss_cell_value
            self.cell_metrics.update(fake_images, nuclei_masks, slide_names)

        if self.perceptual_loss_fn:
            loss_p = self.perceptual_loss_fn.forward(
                fake_images.contiguous(), y.contiguous())
            gen_loss = gen_loss + 0.1 * loss_p

        if self.foreground_head:
            target_foreground = (y > -0.9).type_as(y)
            gen_foreground_loss = self.foreground_loss(
                target=target_foreground, input=foreground_preds)
            gen_loss = gen_loss + gen_foreground_loss

        with torch.no_grad():
            y_clip = y.contiguous()
            fake_images_clip = fake_images.contiguous().clip(-0.9, 0.9)
            getattr(self, f"{prefix}_pix_metrics").update(fake_images_clip, y_clip)

            # if self.foreground_head:
            #    foreground_pred_class = F.sigmoid(foreground_preds) > 0.5
            #    dice_value = self.dice_metric(foreground_pred_class, y_clip > -0.9)

        # Discriminator step
        if self.gan_train:
            disc_output_real = self.discriminator(x, y)
            disc_output = torch.concat([disc_output_fake, disc_output_real], axis=0)
            labels = torch.concat(
                [
                    torch.ones(disc_output_fake.shape).type_as(x),
                    torch.zeros(disc_output_real.shape).type_as(x),
                ],
                axis=0,
            )
            disc_adv_loss = self.adversarial_loss(target=labels, input=disc_output)
            if not self.is_lsgan:
                with torch.no_grad():
                    getattr(self, f"{prefix}_disc_metrics").update(disc_output, labels)
            self.log(f"{prefix}_disc_adv_loss", disc_adv_loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)

        self.log(f"{prefix}_gen_adv_loss", gen_adv_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"{prefix}_gen_loss", gen_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"{prefix}_gen_loss_sim", gen_loss_sim, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        if self.use_cell_metrics and (self.cell_loss is not None):
            self.log(f"{prefix}_loss_cell", loss_cell_value, on_step=False, on_epoch=True,
                     logger=True, prog_bar=True)
        if self.foreground_head:
            self.log(f"{prefix}_gen_foreground_loss", gen_foreground_loss,
                     on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
        if self.perceptual_loss_fn:
            self.log(f"{prefix}_loss_p", loss_p, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Perform a validation step using the evaluation_step method."""
        self.evaluation_step(batch, batch_idx, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Perform a test step using the evaluation_step method."""
        self.evaluation_step(batch, batch_idx, "test")

    def epoch_end_cell_metrics(self, prefix: str, logreg_layer: Optional[nn.Linear] = None,
                               return_dataframe: bool = False):
        """Log and optionally return cell-level classification metrics at the end of an epoch."""
        if return_dataframe:
            cell_metrics, dataframe_cell = self.cell_metrics.compute(logreg_layer, return_dataframe)
        else:
            cell_metrics = self.cell_metrics.compute(logreg_layer)
        self._log_val_metric(f"{prefix}_cell_auc", cell_metrics["auc"])
        self._log_val_metric(f"{prefix}_cell_auc_logreg", cell_metrics["auc_logreg"])
        self._log_val_metric(f"{prefix}_cell_balanced_acc", cell_metrics["balanced_acc"])
        self._log_val_metric(f"{prefix}_cell_f1", cell_metrics["f1"])
        for marker_col in self.cell_metrics.target_names:
            self._log_val_metric(f"{prefix}_cell_auc_{marker_col}",
                                 cell_metrics[f"{marker_col}_auc"])
            self._log_val_metric(f"{prefix}_cell_auc_logreg_{marker_col}",
                                 cell_metrics[f"{marker_col}_auc_logreg"])
            self._log_val_metric(f"{prefix}_cell_balanced_acc_{marker_col}",
                                 cell_metrics[f"{marker_col}_balanced_acc"])
            self._log_val_metric(f"{prefix}_cell_f1_{marker_col}",
                                 cell_metrics[f"{marker_col}_f1"])
        if return_dataframe:
            return cell_metrics, dataframe_cell
        else:
            return cell_metrics

    def on_validation_epoch_end(self):
        """Handle end-of-val-epoch logging, metric resetting, and optional cell metrics update."""
        self.log_dict(self.val_pix_metrics.compute())
        self.log_dict(self.val_disc_metrics.compute())
        self.val_pix_metrics.reset()
        self.val_disc_metrics.reset()

        if self.use_cell_metrics:
            cell_metrics = self.epoch_end_cell_metrics("val")
            self.logreg_layer.load_state_dict(cell_metrics["state_dict"])
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        """Handle end-of-test-epoch logging, metric resetting, and optional cell metrics update."""
        self.log_dict(self.test_pix_metrics.compute())
        self.log_dict(self.test_disc_metrics.compute())
        self.test_pix_metrics.reset()
        self.test_disc_metrics.reset()

        if self.use_cell_metrics:
            _, dataframe_cell = self.epoch_end_cell_metrics(
                "test", logreg_layer=self.logreg_layer, return_dataframe=True)
            if hasattr(self.trainer, "ckpt_path"):
                dataframe_cell_path = str(
                    Path(self.trainer.ckpt_path).parent / "test_dataframe_cell.csv")
                dataframe_cell.to_csv(dataframe_cell_path, index=False)
        super().on_test_epoch_end()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers of Lightning Module."""
        """
        # DEPRECATED
        if self.vit_lr_decay:
            decay_func = functools.partial(get_vit_lr_decay_rate,
                                           num_layers=len(self.generator.encoder.model.blocks),
                                           lr_decay_rate=0.65)
            g_params = []
            for name, param in self.generator.named_parameters():
                if param.requires_grad:
                    # Decay the learning rate for each layer
                    lr = self.lr_g * decay_func(name)
                    g_params.append({'params': param, 'lr': lr})
            g_optimizer = torch.optim.Adam(g_params, betas=(0.5, 0.999), eps=1e-7)
        else:
            g_optimizer = torch.optim.Adam(
                self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999), eps=1e-7)
        """
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999), eps=1e-7)
        total_iters = self.trainer.estimated_stepping_batches
        g_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                g_optimizer,
                lr_lambda=pix2pix_lr_scheduler(total_iters, 400, total_iters // 2)
            ),
            'interval': 'step',  # 'epoch' or 'step'
            'frequency': 1
        }
        if self.gan_train:
            d_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999), eps=1e-7)
            d_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    d_optimizer,
                    lr_lambda=pix2pix_lr_scheduler(total_iters, 400, total_iters // 2)
                ),
                'interval': 'step',  # 'epoch' or 'step'
                'frequency': 1
            }
            return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]
        else:
            d_optimizer = None
            d_scheduler = None
            return [g_optimizer], [g_scheduler]

    def _log_train_metric(self, metric_name: str, metric_value):
        """Log a training metric both per step and per epoch."""
        self.log(metric_name + "_step", metric_value, on_step=True, on_epoch=False,
                 prog_bar=True, logger=False)
        self.log(metric_name, metric_value, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

    def _log_val_metric(self, metric_name: str, metric_value):
        """Log a validation/test metric both per step and per epoch."""
        self.log(metric_name, metric_value, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)


class DiscriminatorPatch(nn.Module):
    """
    Define a PatchGAN discriminator for use in GAN architectures.

    This class implements a PatchGAN discriminator as described in the CycleGAN and pix2pix papers.
    It supports configurable normalization layers, dropout, and channel selection for the generated
    images.
    Args:
        input_nc (int): Number of channels in the input images.
        ndf (int, optional): Number of filters in the last convolutional layer. Defaults to 64.
        n_layers (int, optional): Number of convolutional layers in the discriminator.
            Defaults to 3.
        dropout_rate (float, optional): Dropout rate for Dropout2d layers. Defaults to 0.
        norm_layer_type (Optional[str], optional): Type of normalization layer to use ('batch',
            'instance', or None). Defaults to None.
        selected_channels (Optional[torch.Tensor], optional): Indices of channels to select from
            the generated images if you want to apply the discriminator only on some markers.
            Defaults to None.
    Attributes:
        selected_channels (Optional[torch.Tensor]): Indices of channels to select from the
            generated images if you want to apply the discriminator only on some markers.
        model (nn.Sequential): The sequential model representing the PatchGAN discriminator.
    Methods:
        forward(x, fake_images):
            Performs a forward pass through the discriminator, optionally selecting specific
                channels from the generated images and concatenating them with the input images.
        _weights_init(m):
            Initializes the weights of the model's layers.
    References:
        - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3,
                 dropout_rate: float = 0., norm_layer_type: Optional[str] = None,
                 selected_channels: Optional[torch.Tensor] = None):
        super(DiscriminatorPatch, self).__init__()
        self.selected_channels = selected_channels
        if norm_layer_type == "batch":
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_layer_type == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
        elif norm_layer_type is None:
            norm_layer = nn.Identity
        else:
            raise ValueError("norm_layer_type should be batch or instance")
        use_bias = norm_layer == nn.Identity

        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, inplace=False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                                        padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout2d(dropout_rate),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                                    padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout2d(dropout_rate),
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                             stride=1, padding=padw))]
        self.model = nn.Sequential(*sequence)
        self.model.apply(self._weights_init)

    def _weights_init(self, m: nn.Module) -> None:
        """Initialize weights of the model."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            # nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel selection on generated image.

        Parameters:
            x (tensor)          -- RGB input image
            fake_images (tensor)-- Generated image

        Returns:
            Tensor              -- Discriminator output
        """
        # Select channels from generated image if indices are specified
        if self.selected_channels is not None:
            fake_images = torch.index_select(fake_images, dim=1, index=self.selected_channels)

        # Concatenate RGB input and selected generated channels
        input = torch.cat([x, fake_images], dim=1)

        return self.model(input)
