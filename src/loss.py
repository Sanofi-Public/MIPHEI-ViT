"""Loss functions for training MIPHEI-VIT."""

from typing import Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error (MSE) Loss for multi-channel data with marker-specific weighting.

    This loss function is proposed in our paper to ensure equal importance is given to each marker,
    regardless of their abundance or distribution differences. Each channel (marker) is weighted by
    the inverse of the variance (or something different) of its full distribution (including
    background, not only positive signal), compensating for varying marker abundances.
    Args:
        lambda_factor (float): Scaling factor applied to the final loss. Intended to match Pix2pix
            reconstruction loss.
        marker_weights (Tensor): 1D tensor of per-marker weights, typically the inverse of the
            variance for each channel.
    Attributes:
        lambda_factor (float): Scaling factor for the loss (Mimic Pix2Pix reconstruction loss).
        marker_weights (Tensor): Buffer containing per-marker weights.
    Forward Args:
        y_true (Tensor): Ground truth tensor of shape (batch_size, num_markers, height, width).
        y_pred (Tensor): Predicted tensor of the same shape as y_true.
    Returns:
        Tensor: Scalar weighted MSE loss value.
    """

    def __init__(self, lambda_factor: float, marker_weights: torch.Tensor):
        super(WeightedMSELoss, self).__init__()
        self.lambda_factor = lambda_factor
        self.register_buffer("marker_weights", marker_weights)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the weighted mean squared error loss between predictions and targets."""
        loss = F.mse_loss(target=y_true, input=y_pred, reduction="none")
        loss = loss.mean(dim=(0, 2, 3)) * self.marker_weights
        return loss.mean() * self.lambda_factor


def get_focal_loss(lambda_factor: float, foreground_weight: torch.Tensor) -> Callable:
    """
    Return a focal loss function for regression tasks using L3 loss.

    This implementation computes a focal loss by raising the element-wise L1 loss to the power of 3
    (L3 loss), and applies class balancing using the provided foreground weights. The final loss is
    scaled by the given lambda factor (Pix2pix style reconstruction loss).

    Args:
        lambda_factor (float): Scaling factor to adjust the overall magnitude of the loss. (Pix2pix
            style reconstruction loss)
        foreground_weight (Tensor): 1D tensor of weights for each class or label, used to balance
            the loss contribution from different classes. Can be the variance.
    Returns:
        Callable: A loss function that takes (y_true, y_pred) as input and returns the computed
            focal loss.
    """
    marker_weights = foreground_weight / foreground_weight.sum()

    def focal_loss(y_true, y_pred):
        focal_loss = F.l1_loss(target=y_true, input=y_pred, reduction="none") ** 3
        focal_loss = (focal_loss * marker_weights).sum(dim=1).mean()
        return focal_loss * lambda_factor
    return focal_loss


def get_shrinkage_loss(lambda_factor: float, foreground_weight: torch.Tensor) -> Callable:
    """
    Create a shrinkage loss function with label weighting and a non-linear penalty.

    The returned loss function computes a modified L1 loss with a non-linear shrinkage term,
    applies channel-wise weights, and scales the result by `lambda_factor`.
    Args:
        lambda_factor (float): Scaling factor to control the overall strength of the loss (Pix2pix
            style reconstruction loss).
        foreground_weight (torch.Tensor): 1D tensor of weights for each label, used to compute
            channel-wise weighting.
    Returns:
        Callable: A loss function that takes (y_true, y_pred) as input and returns a scalar loss
            value.
    """
    marker_weights = foreground_weight / foreground_weight.sum()

    def shrinkage_loss(y_true, y_pred):
        # return torch.nn.functional.l1_loss(target=y_true, input=y_pred) * lambda_factor
        l1 = torch.abs(y_true - y_pred)
        loss = l1**2 / (1 + torch.exp(10 * (0.2 - l1)))
        loss = (loss * marker_weights).sum(dim=1).mean()
        return loss * lambda_factor
    return shrinkage_loss


class L1_L2_Loss(nn.Module):
    """
    Loss function that combines L1 (Mean Absolute Error) and L2 (Mean Squared Error) losses.

    This loss computes the average of L1 and L2 losses between the predicted and true values,
    scaled by a user-defined lambda factor (Pix2pix style reconstruction loss).
    Args:
        lambda_factor (float): Scaling factor to control the contribution of the combined loss.
    Methods:
        forward(y_pred, y_true):
            Computes the combined L1 and L2 loss between predictions and targets.
    Example:
        loss_fn = L1_L2_Loss(lambda_factor=0.5)
        loss = loss_fn(y_pred, y_true)
    """

    def __init__(self, lambda_factor: float):
        super().__init__()
        self.lambda_factor = lambda_factor
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the weighted average of L1 and L2 losses between predictions and targets."""
        return self.lambda_factor * (self.l1_loss(
            input=y_pred, target=y_true) + self.l2_loss(input=y_pred, target=y_true)) / 2


def get_mae_loss(lambda_factor: float) -> Callable:
    """
    Mean absolute error (MAE) loss scaled by a lambda factor.

    The returned loss function computes the MAE between the true and predicted values,
    and multiplies the result by the specified lambda factor. This is useful for
    weighting the MAE loss in a composite loss function (Pix2pix style reconstruction loss).
    Args:
        lambda_factor (float): A scaling factor to multiply the MAE loss.

    Returns:
        Callable: A loss function that takes (y_true, y_pred) as arguments and returns the scaled
            MAE loss.
    """
    def mae_loss(y_true, y_pred):
        return F.l1_loss(target=y_true, input=y_pred) * lambda_factor
    return mae_loss


def get_mse_loss(lambda_factor: float) -> Callable:
    """
    Mean squared error (MSE) loss function scaled by a lambda factor.

    The returned loss function computes the MSE between the true and predicted values,
    and multiplies the result by the specified lambda factor. This is useful for
    weighting the MSE loss in a composite loss function.

    Args:
        lambda_factor (float): A scaling factor to weight the MSE loss.

    Returns:
        Callable: A function that takes (y_true, y_pred) as arguments and returns the scaled
            MSE loss.
    """
    def mse_loss(y_true, y_pred):
        return F.mse_loss(target=y_true, input=y_pred) * lambda_factor
    return mse_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.

    Focal Loss is designed to address class imbalance by down-weighting easy examples and focusing
    training on hard negatives. This implementation uses the formulation from the RetinaNet paper.
    Args:
        alpha (float, optional): Weighting factor for the rare class. Default is 0.25.
        gamma (float, optional): Focusing parameter that reduces the relative loss for
            well-classified examples. Default is 2.
    Forward Args:
        input (torch.Tensor): Predicted logits of shape (N, *).
        target (torch.Tensor): Ground truth binary labels of the same shape as input.
    Returns:
        torch.Tensor: Scalar tensor representing the focal loss.
    Paper:
        - Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense
            Object Detection. https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss between input and target tensors."""
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedBCEAndDiceLoss:
    """
    Combine Binary Cross Entropy (BCE) loss with Dice loss for segmentation tasks.

    This loss function is useful for binary segmentation problems, especially when dealing with
    class imbalance. The BCE component uses a foreground weight to emphasize positive (foreground)
    examples, while the Dice loss measures overlap between predicted and ground truth masks. For
    H&E to mIF it can be used for foreground (positive protein signal) segmentation.
    Args:
        foreground_weight (float, optional): Weight for the positive class in BCE loss.
            Default is 1.0.
    Attributes:
        foreground_weight (float): Weight for the positive class in BCE loss.
        bce_loss (nn.BCEWithLogitsLoss): BCE loss function with specified positive class weight.
    Methods:
        dice_loss(y_pred, y_true):
            Computes the Dice loss between predictions and targets.
        __call__(y_pred, y_true):
            Computes the combined BCE and Dice loss.
    Example:
        >>> loss_fn = CombinedBCEAndDiceLoss(foreground_weight=2.0)
        >>> loss = loss_fn(y_pred, y_true)
    """

    def __init__(self, foreground_weight: float = 1.0):
        super(CombinedBCEAndDiceLoss, self).__init__()
        self.foreground_weight = foreground_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(foreground_weight))

    def dice_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss between predicted logits and ground truth labels."""
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(y_pred)
        # Calculate Dice coefficient
        num = 2 * (probs * y_true).sum() + 1e-5
        den = probs.sum() + y_true.sum() + 1e-5
        dice = num / den
        # Dice loss
        return 1 - dice

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the combined BCE and Dice loss for the given predictions and targets."""
        # Adjusting BCE loss calculation to incorporate foreground_weight for positive targets
        # pos_weight is used to increase the loss for positive examples (foreground)
        bce_loss = self.bce_loss(y_pred, y_true)
        # Dice loss remains unaffected by foreground_weight directly
        dice_loss = self.dice_loss(y_pred, y_true)
        # Sum of BCE and Dice losses as the combined loss
        combined_loss = bce_loss + dice_loss
        return combined_loss


# --------------------------------------------------------------------------------------------------
# The following classes are not used in the main training loop, but are included for reference and
# exploratory purposes.
# --------------------------------------------------------------------------------------------------


class CellLoss(nn.Module):  # not used
    """
    Cell-level loss combining MSE and clustering-based losses on cell mean expressions.

    This loss can be used to guide training with cell-level supervision by encouraging the
    model to reconstruct accurate cell expressions. We can expect a boost on cell level evaluation.
    The loss operates on mean predicted intensity within each label instance (nuclei mask),
    which is obtained from `MeanCellExtrator` wich is a differentiable operation that allows
    gradients to flow from the cell-level loss back to the network. The loss supports two
    components:
      - Mean Squared Error (MSE) loss between predicted and target cell means.
      - Clustering-based loss using a learned MLP to encourage clustering of cell representations.
    Args:
        mlp_path (str): Path to the pretrained MLP model used for clustering loss.
        n_channels (int): Number of channels in the input data.
        use_mse (bool, optional): Whether to include MSE loss. Defaults to True.
        use_clustering (bool, optional): Whether to include clustering loss. Defaults to True.
        lambda_factor (float, optional): Weighting factor for the MSE loss. Defaults to 50.
    Forward Args:
        pred_cell_means (Tensor): Predicted mean intensities for each cell instance.
        target_cell_means (Tensor): Ground truth mean intensities for each cell instance.
    Returns:
        Tensor: Combined loss value (scalar).
    Warning: Only an exploratory idea, not yet tested in practice. Need deeper investigation.
    """

    def __init__(self, mlp_path: str, n_channels: int, use_mse: bool = True,
                 use_clustering: bool = True, lambda_factor: float = 50):
        super(CellLoss, self).__init__()
        self.lambda_factor = lambda_factor
        self.use_mse = use_mse
        self.use_clustering = use_clustering
        self._use_loss = self.use_mse or self.use_clustering
        if use_clustering:
            self.clustering_loss = CellClassificationLoss(mlp_path, n_channels)

    def forward(self, pred_cell_means: torch.Tensor, target_cell_means: torch.Tensor
                ) -> torch.Tensor:
        """Compute the cell loss between predicted and target cell means."""
        if (pred_cell_means.numel() == 0) or (not self._use_loss):
            return 0.
        else:
            if self.use_clustering:
                pred_cell_means_unorm = (pred_cell_means + 0.9) / 1.8 * 255
                target_cell_means_unorm = (target_cell_means + 0.9) / 1.8 * 255
                loss_cluster = self.clustering_loss(pred_cell_means_unorm, target_cell_means_unorm)
            else:
                loss_cluster = 0.
            if self.use_mse:
                loss_mse = F.mse_loss(pred_cell_means, target_cell_means)
            else:
                loss_mse = 0.
            loss = loss_mse * self.lambda_factor + loss_cluster
            return loss


class CellClassificationLoss(nn.Module):  # not used
    """
    Custom loss function for cell classification tasks using a pre-trained MLP and focal loss.

    This loss module applies a pre-trained multi-layer perceptron (MLP) to both input and
    target tensors, producing probabilistic outputs. The loss is computed using the FocalLoss
    criterion between the predicted probabilities and binarized target probabilities. The MLP
    weights are loaded from a checkpoint file.
    Attributes:
        mlp (nn.Sequential): The pre-trained MLP model used to process input and target tensors.
        eps (float): Small value to clamp probabilities for numerical stability.
        criterion (FocalLoss): Focal loss function for robust classification.
    Args:
        mlp_path (str): Path to the checkpoint file containing the pre-trained MLP weights.
        n_channels (int): Number of input channels/features for the MLP.
    Methods:
        forward(input, target):
            Computes the loss between the processed input and target tensors.
    Example:
        loss_fn = CellClassificationLoss(mlp_path="path/to/mlp.pth", n_channels=128)
        loss = loss_fn(input_tensor, target_tensor)
    """

    def __init__(self, mlp_path: str, n_channels: int):
        super(CellClassificationLoss, self).__init__()
        self.mlp = nn.Sequential(
            NormalizationLayer(n_channels),
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_channels - 1),
            nn.Sigmoid()
        )
        state_dict = torch.load(mlp_path)["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.mlp.load_state_dict(state_dict)
        self.mlp.eval()
        self.eps = 1e-6
        self.criterion = FocalLoss(alpha=0.5, gamma=2)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between processed input and target tensors."""
        prob_input = self.mlp(input).clamp(self.eps, 1.0 - self.eps)
        with torch.no_grad():
            prob_target = self.mlp(target).clamp(self.eps, 1.0 - self.eps)
        """kl_div_per_class = prob_target * torch.log(prob_target / prob_input) + \
                       (1 - prob_target) * torch.log((1 - prob_target) / (1 - prob_input))
        # Average across classes and batch
        loss = torch.mean(torch.sum(kl_div_per_class, dim=1))"""
        # loss = F.mse_loss(prob_input, prob_target)
        loss = self.criterion(input=prob_input, target=(prob_target > 0.5).to(prob_target.dtype))
        return loss


class NormalizationLayer(nn.Module):
    """
    A PyTorch module for normalizing input data using specified mean and standard deviation.

    This layer subtracts the mean and divides by the standard deviation for each channel,
    which is useful for preprocessing input data before feeding it into a neural network.
    Attributes:
        mean (torch.Tensor): The mean value(s) for normalization, stored as a buffer.
        std (torch.Tensor): The standard deviation value(s) for normalization, stored as a buffer.
    Args:
        n_channels (int): Number of channels in the input data.
        mean (list or None, optional): Mean values for each channel. If None, defaults to zeros.
        std (list or None, optional): Standard deviation values for each channel.
            If None, defaults to ones.
    Example:
        >>> norm = NormalizationLayer(3, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        >>> x = torch.randn(1, 3, 224, 224)
        >>> x_norm = norm(x)
    """

    def __init__(self, n_channels: int, mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None):
        super(NormalizationLayer, self).__init__()
        if mean is None:
            mean = [0.] * n_channels
        if std is None:
            std = [1.] * n_channels
        self.register_buffer("mean", torch.tensor(mean).flatten())
        self.register_buffer("std", torch.tensor(std).flatten())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor x using stored mean and std."""
        return (x - self.mean) / self.std
