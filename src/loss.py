"""Loss functions for training the model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_weighted_mae_loss(sim_loss_factor, foreground_weight, foreground_thresh):
    ones_weight = torch.ones_like(foreground_weight)
    #background_weight = 1 - foreground_weight
    def mae_loss(y_true, y_pred):
        foreground_mask = y_true > foreground_thresh
        loss_weights = torch.where(foreground_mask, foreground_weight, ones_weight)
        #loss_weights = torch.where(y_true > foreground_thresh, foreground_weight, background_weight)
        mae_loss = torch.mean(torch.nn.functional.l1_loss(
            y_pred, y_true, reduction="none") * loss_weights)
        #return sim_loss_factor * mae_loss
        return 2 * sim_loss_factor * mae_loss
    return mae_loss


def get_mae_loss(lambda_factor):
    def mae_loss(y_true, y_pred):
        return F.l1_loss(target=y_true, input=y_pred) * lambda_factor
    return mae_loss


def get_mse_loss(lambda_factor):
    def mse_loss(y_true, y_pred):
        return F.mse_loss(target=y_true, input=y_pred) * lambda_factor
    return mse_loss


class WeightedMSELoss(nn.Module):
    """Normalize data using mean and std."""
    def __init__(self, lambda_factor, marker_weights):
        super(WeightedMSELoss, self).__init__()
        self.lambda_factor = lambda_factor
        self.register_buffer("marker_weights", marker_weights)
    
    def forward(self, y_true, y_pred):
        loss = F.mse_loss(target=y_true, input=y_pred, reduction="none")
        loss = loss.mean(dim=(0, 2, 3)) * self.marker_weights
        return loss.mean() * self.lambda_factor


def get_focal_loss(lambda_factor, foreground_weight):
    label_weights = foreground_weight / foreground_weight.sum()
    def focal_loss(y_true, y_pred):
        focal_loss = F.l1_loss(target=y_true, input=y_pred, reduction="none") ** 3
        focal_loss = (focal_loss * label_weights).sum(dim=1).mean()
        return focal_loss * lambda_factor 
    return focal_loss


def get_shrinkage_loss(lambda_factor, foreground_weight):
    label_weights = foreground_weight / foreground_weight.sum()
    def shrinkage_loss(y_true, y_pred):
        #return torch.nn.functional.l1_loss(target=y_true, input=y_pred) * lambda_factor
        l = torch.abs(y_true - y_pred)
        loss = l**2 / (1 + torch.exp(10 * (0.2 - l)))
        loss = (loss * label_weights).sum(dim=1).mean()
        return loss * lambda_factor
    return shrinkage_loss


def compute_image_gradients(image):
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    
    # Apply the Sobel filters to get gradients in x and y directions
    grad_x = torch.nn.functional.conv2d(
        image, sobel_x, padding=1, groups=image.shape[1])  # Gradient in x direction
    grad_y = torch.nn.functional.conv2d(
        image, sobel_y, padding=1, groups=image.shape[1])  # Gradient in y direction
    
    return grad_x, grad_y

def structural_loss(generated, target):
    gen_grad_x, gen_grad_y = compute_image_gradients(generated)
    with torch.no_grad():
        tgt_grad_x, tgt_grad_y = compute_image_gradients(target)
    
    edge_loss = torch.nn.functional.l1_loss(
        gen_grad_x, tgt_grad_x) + torch.nn.functional.l1_loss(gen_grad_y, tgt_grad_y)
    
    return edge_loss


def total_variation_loss(image):
    # Calculate difference between adjacent pixels in x and y directions
    tv_loss_x = torch.nn.functional.l1_loss(
        image[:, :, :, :-1], image[:, :, :, 1:])
    tv_loss_y = torch.nn.functional.l1_loss(
        image[:, :, :-1, :], image[:, :, 1:, :])
    return tv_loss_x + tv_loss_y


class L1_L2_Loss(nn.Module):
    # Learned perceptual metric
    def __init__(self, lambda_factor):
        super().__init__()
        self.lambda_factor = lambda_factor
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.lambda_factor * (self.l1_loss(
            input=y_pred, target=y_true) + self.l2_loss(input=y_pred, target=y_true)) / 2


class CombinedBCEAndDiceLoss:
    def __init__(self, foreground_weight=1.0):
        super(CombinedBCEAndDiceLoss, self).__init__()
        self.foreground_weight = foreground_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(foreground_weight))

    def dice_loss(self, y_pred, y_true):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(y_pred)
        # Calculate Dice coefficient
        num = 2 * (probs * y_true).sum() + 1e-5
        den = probs.sum() + y_true.sum() + 1e-5
        dice = num / den
        # Dice loss
        return 1 - dice

    def __call__(self, y_pred, y_true):
        # Adjusting BCE loss calculation to incorporate foreground_weight for positive targets
        # pos_weight is used to increase the loss for positive examples (foreground)
        bce_loss = self.bce_loss(y_pred, y_true)
        # Dice loss remains unaffected by foreground_weight directly
        dice_loss = self.dice_loss(y_pred, y_true)
        # Sum of BCE and Dice losses as the combined loss
        combined_loss = bce_loss + dice_loss
        return combined_loss


class CellLoss(nn.Module):
    def __init__(self, mlp_path, n_channels, use_mse=True, use_clustering=True, lambda_factor=50):
        super(CellLoss, self).__init__()
        self.lambda_factor = lambda_factor
        self.use_mse = use_mse
        self.use_clustering = use_clustering
        self._use_loss = self.use_mse or self.use_clustering
        if use_clustering:
            self.clustering_loss = CellClusterLoss(mlp_path, n_channels)

    def forward(self, pred_cell_means, target_cell_means):

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


class CellClusterLoss(nn.Module):
    def __init__(self, mlp_path, n_channels):
        super(CellClusterLoss, self).__init__()
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

    def forward(self, input, target):
        prob_input = self.mlp(input).clamp(self.eps, 1.0 - self.eps)
        with torch.no_grad():
            prob_target = self.mlp(target).clamp(self.eps, 1.0 - self.eps)
        """kl_div_per_class = prob_target * torch.log(prob_target / prob_input) + \
                       (1 - prob_target) * torch.log((1 - prob_target) / (1 - prob_input))
        # Average across classes and batch
        loss = torch.mean(torch.sum(kl_div_per_class, dim=1))"""
        #loss = F.mse_loss(prob_input, prob_target)
        loss = self.criterion(input=prob_input, target=(prob_target > 0.5).to(prob_target.dtype))
        return loss


class NormalizationLayer(nn.Module):
    """Normalize data using mean and std."""
    def __init__(self, n_channels, mean=None, std=None):
        super(NormalizationLayer, self).__init__()
        if mean is None:
            mean = [0.] * n_channels
        if std is None:
            std = [1.] * n_channels
        self.register_buffer("mean", torch.tensor(mean).flatten())
        self.register_buffer("std", torch.tensor(std).flatten())

    def __call__(self, x):
        return (x - self.mean) / self.std
