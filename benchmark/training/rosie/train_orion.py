"""
Training script for H&E to multiplex protein prediction model.

This script trains a deep learning model to predict protein expression levels
from H&E images. It supports both training and evaluation modes.

Required directory structure:
ROOT_DIR/
    ├── data/                     # Contains training data
    │   └── cell_measurements.pqt # Parquet file with cell measurements
    ├── images/                   # H&E image data
    │   └── {uuid}/image.ome.zarr # Zarr formatted image files  
    ├── metadata/                 # Metadata files
    │   └── metadata_dict.pkl     # Dictionary with experiment metadata
    └── runs/                     # Training run outputs
"""

import pyvips

import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import pandas as pd
import wandb
from typing import Tuple, List, Dict, Optional
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import torch.nn.functional as F
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Configure torch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Configuration constants
DATA_FILE = "/root/workdir/ORION_dataset_20x/dataframe_tile.csv"
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
OUTPUT_DIR = "/root/workdir/rosie/runs"
EXPERIMENT_NAME = "rosie_orion"

# Model training constants
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EVAL_INTERVAL = 3000
PATIENCE = 75000
NUM_WORKERS = 10 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PATCH_SIZE = 128

# Dataset splits for train/val/test
DATASET_SPLITS = {
    'train': [
        '19510_C8_US_SCAN_OR_001__150825-registered.ome',
        '19510_C26_US_SCAN_OR_001__092131-registered.ome',
        '19510_C17_US_SCAN_OR_001__152525-registered.ome',
        '19510_C35_US_SCAN_OR_001__161209-registered.ome'
        '19510_C39_US_SCAN_OR_001__162343-registered.ome',
        '19510_C13_US_SCAN_OR_001__151503-registered.ome',
        '19510_C31_US_SCAN_OR_001__160203-registered.ome',
        '19510_C22_US_SCAN_OR_001__092420-registered.ome',
        '18459_LSP10388_US_SCAN_OR_001__091155-registered.ome',
        '19510_P37-S83_C40_US_SCAN_OR_001__163912-registered.ome',
        '19510_C28_US_SCAN_OR_001__155413-registered.ome',
        '19510_C37_US_SCAN_OR_001__161733-registered.ome',
        '19510_C20_US_SCAN_OR_001__153341-registered.ome',
        '19510_C15_US_SCAN_OR_001__152234-registered.ome',
        '19510_C33_US_SCAN_OR_001__160715-2-registered.ome',
        '19510_C24_US_SCAN_OR_001__091904-registered.ome',
        '18459_LSP10408_US_SCAN_OR_001__092559-registered.ome',
        '18459_LSP10441_US_SCAN_OR_001__091844-registered.ome',
        '19510_C27_US_SCAN_OR_001__155205-registered.ome',
        '19510_C18_US_SCAN_OR_001__152757-registered.ome',
        '19510_C36_US_SCAN_OR_001__161442-registered.ome',
        '18459_LSP10452_US_SCAN_OR_001__091355-registered.ome',
        '18459_LSP10353_US_SCAN_OR_001__093059-registered.ome',
        '19510_C14_US_SCAN_OR_001__151737-registered.ome',
        '19510_C32_US_SCAN_OR_001__160434-registered.ome',
        '19510_C23_US_SCAN_OR_001__154147-registered.ome',
        '18459_LSP10397_US_SCAN_OR_001__091631-registered.ome',
        '19510_C29_US_SCAN_OR_001__155859-registered.ome',
        '19510_C38_US_SCAN_OR_001__162018-registered.ome',
        '19510_C12_US_SCAN_OR_001__151249-registered.ome',
        '19510_C21_US_SCAN_OR_001__153607-registered.ome',
        '18459_LSP10375_US_SCAN_OR_001__092147-registered.ome',
        '19510_C16_US_SCAN_OR_001__152020-registered.ome',
        '19510_C33_US_SCAN_OR_001__160715-registered.ome',
        '19510_C34_US_SCAN_OR_001__160949-registered.ome',
        '19510_C25_US_SCAN_OR_001__154712-registered.ome',
        '18459_LSP10419_US_SCAN_OR_001__090907-registered.ome'
    ],
    'val': [
        '19510_C19_US_SCAN_OR_001__153041-registered.ome',
        '19510_C30_US_SCAN_OR_001__155702-registered.ome'
    ],
    'test': [
        '19510_C11_US_SCAN_OR_001__151039-registered.ome',
        '18459_LSP10364_US_SCAN_OR_001__092347-registered.ome'
    ]
}

def pad_patch(patch: np.ndarray, 
             original_size: Tuple[int, int], 
             x_center: int, 
             y_center: int, 
             patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Pads the given patch if its size is less than patch_size x patch_size pixels.

    Args:
        patch: NumPy array representing the patch image
        original_size: Tuple of (width, height) of the original image
        x_center: X coordinate of the center of the patch in the original image
        y_center: Y coordinate of the center of the patch in the original image
        patch_size: The target size of the patch

    Returns:
        Padded patch as a NumPy array
    """
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]
    
    if current_height == patch_size and current_width == patch_size:
        return patch
        
    # Calculate padding needed
    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)

    # Apply padding
    pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if patch.ndim == 3 else ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=0)

    # Ensure the patch is exactly patch_size x patch_size
    padded_patch = padded_patch[:patch_size, :patch_size]

    return padded_patch

def masked_mse_loss(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss with a mask.

    Args:
        pred: Predicted tensor
        target: Target tensor
        mask: Mask tensor with 1s for elements to include and 0s to exclude

    Returns:
        Loss value
    """
    mask = mask.bool()
    masked_pred = torch.masked_select(pred, mask)
    masked_target = torch.masked_select(target, mask)
    return F.mse_loss(masked_pred, masked_target, reduction='mean')

def get_model(num_outputs: Optional[int] = None, 
             use_context: bool = False, 
             use_mask: bool = False) -> nn.Module:
    """
    Creates and returns the model architecture.

    Args:
        num_outputs: Number of output features to predict
        use_context: Whether to use contextual features
        use_mask: Whether to use masking in the model

    Returns:
        PyTorch model instance
    """
    model = models.convnext_small(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model

class OrionImageDataset(Dataset):
    """
    Dataset class for loading H&E image patches and their corresponding protein expression values.
    
    Args:
        data_df: DataFrame containing cell measurements
        is_test: Whether this is a test dataset
        use_mask: Whether to use cell segmentation masks
        transform: Transforms to apply to images
        all_biomarkers: List of all biomarker names
        subset: Subset of Slide Names to use
    """
    def __init__(self,
                data_df: pd.DataFrame,
                is_test: bool = False,
                transform: Optional[Dict] = None,
                all_biomarkers: Optional[List] = [],
                subset: Optional[List[str]] = None):
        
        self.df = data_df
        self.df["tile_name"] = self.df["image_path"].apply(lambda x: Path(x).stem)
        # Map each unique 'in_slide_name' to an integer index
        unique_slides = self.df['in_slide_name'].unique()
        slide_to_idx = {slide: idx for idx, slide in enumerate(unique_slides)}
        self.df['seg_acq_id'] = self.df['in_slide_name'].map(slide_to_idx)

        self.transform = transform
        self.patch_size = PATCH_SIZE
        self.ps = self.patch_size//2
        self.all_biomarkers = all_biomarkers
        self.invalid_acq_ids = set()
        self.zarr_cache = {}
        self.is_test = is_test

        assert len(self.all_biomarkers) != 0, "No biomarker labels found"
        
        if subset is not None:
            self.df = self.df[self.df['in_slide_name'].isin(subset)]
        self.df.reset_index(inplace=True)
        self.acq_map = {i: x for i, x in enumerate(self.df['seg_acq_id'].unique())}
        self.acq_map.update({x: i for i, x in enumerate(self.df['seg_acq_id'].unique())})

        self.df = pd.concat([self.df] * 6, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def get_crop_tissue_positions(tissue_mask, rng):
        ys, xs = np.where(tissue_mask)
        idx = rng.randint(ys.size)
        y, x = int(ys[idx] + 64), int(xs[idx] + 64)
        return x, y

    def __getitem__(self, idx: int) -> Optional[Tuple]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple containing (image_patch, expression_values, mask, metadata)
            Returns None if the item is invalid
        """
        if self.is_test:
            rng = np.random.RandomState(idx)
        else:
            rng = np.random

        row = self.df.iloc[idx]
        seg_acq_id = row['seg_acq_id']
        tile_name = row['tile_name']
        he_region_path = row['image_path']
        if (not self.is_test) and (np.random.uniform() > 0.25):
            he_parent = Path(he_region_path).parent.parent / "he_augmented"
            he_region_path = str(he_parent / Path(he_region_path).name)
        if_region_path = row['target_path']
        X, Y = list(map(int, tile_name.split("_")[-5:-3]))

        # Read H&E and mIF patches
        he_patch = Image.open(he_region_path)
        if_patch = pyvips.Image.new_from_file(if_region_path)[self.all_biomarkers]

        # Random Crop to 128x128 where there is signal
        roi_mask = if_patch.crop(self.ps, self.ps, if_patch.height - self.patch_size,
                                 if_patch.width - self.patch_size).bandmean().numpy() > 1
        if rng.uniform() < 0.05:  # takes in background with prob 5%
                roi_mask = ~roi_mask
        if not roi_mask.any():  # no mIF signal
            x = rng.randint(self.ps, if_patch.height - self.ps)
            y = rng.randint(self.ps, if_patch.width - self.ps)
        else:
            x, y = self.get_crop_tissue_positions(roi_mask, rng)

        X += x
        Y += y
        he_patch = he_patch.crop((x - self.ps, y - self.ps, x + self.ps, y + self.ps))  # 128x128
        if_patch = if_patch.crop(x - 4, y - 4, 8, 8) # 8x8

        # Handle expression values
        exp_vec = np.float32(if_patch.numpy().mean(axis=(0, 1))) / 255
        valid_mask = np.ones(len(self.all_biomarkers))

        # Apply transforms
        seed = rng.randint(2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if isinstance(self.transform, dict):
            he_patch_pt = self.transform['all_channels'](he_patch)
            patch = self.transform['image_only'](he_patch_pt)
        else:
            patch = self.transform(he_patch)
        assert patch.shape == (3, 224, 224), f'Patch shape is {patch.shape}'

        return patch, exp_vec, valid_mask, X, Y, seg_acq_id


def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device,
            run_dir: str,
            step: int,
            bm_labels: List[str],
            acq_dict: Optional[Dict] = None,
            save_predictions: bool = False,
            pred_biomarkers: Optional[List[str]] = None) -> Optional[Tuple[float, float]]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        run_dir: Directory to save results
        step: Current training step
        bm_labels: List of biomarker labels
        acq_dict: Dictionary mapping acquisition IDs to metadata
        save_predictions: Whether to save predictions to disk
        pred_biomarkers: List of biomarkers to predict (if different from bm_labels)
        
    Returns:
        None
    """
    model.eval()
    if pred_biomarkers is None:
        pred_biomarkers = bm_labels
        
    try:
        acq_map = data_loader.dataset.acq_map
    except:
        acq_map = data_loader.dataset.dataset.acq_map
    
    eval_dataset = []
    save_interval = 2000
    
    with torch.no_grad():
        for idx, (inputs, exp_vec, mask, X, Y, indices) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs).detach().cpu().numpy()
            indices = indices.numpy()
            exp_vec = exp_vec.numpy()
            
            acq_ids = [acq_map[x] for x in indices]
            rows = [[a,b,c]+list(d)+list(e) for a,b,c,d,e in zip(X.numpy(), Y.numpy(), acq_ids, outputs, exp_vec)]
            eval_dataset.extend(rows)
            
            if save_predictions and idx % save_interval == 0:
                temp_df = pd.DataFrame(eval_dataset, 
                                     columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] + 
                                             [f'pred_{x}' for x in pred_biomarkers] + 
                                             [f'gt_{x}' for x in bm_labels])
                
                if os.path.exists(f'{run_dir}/predictions_{step}_{idx}.pqt'):
                    temp_df.to_parquet(f'{run_dir}/predictions_{step}_{idx}.pqt', 
                                     engine='fastparquet', append=True)
                else:
                    temp_df.to_parquet(f'{run_dir}/predictions_{step}_{idx}.pqt')
                eval_dataset = []

    eval_dataset = pd.DataFrame(eval_dataset, 
                              columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] + 
                                      [f'pred_{x}' for x in pred_biomarkers] + 
                                      [f'gt_{x}' for x in bm_labels])

    if save_predictions:
        eval_dataset.to_parquet(f'{run_dir}/predictions_{step}.pqt')

    pred_cols = [col for col in eval_dataset.columns if col.startswith('pred_')]
    gt_cols = [col for col in eval_dataset.columns if col.startswith('gt_')]
    
    pearson_r = 0.
    spearman_r = 0.
    r2 = 0.
    for pred_col, gt_col in zip(pred_cols, gt_cols):
        y_true_all = eval_dataset[gt_col].to_numpy()
        y_pred_all = eval_dataset[pred_col].to_numpy()
        pearson_r += pearsonr(y_true_all, y_pred_all)[0]
        spearman_r += spearmanr(y_true_all, y_pred_all)[0]
        r2 += r2_score(y_true_all, y_pred_all)
    pearson_r /= len(pred_cols)
    spearman_r /= len(pred_cols)
    r2 /= len(pred_cols)

    return {'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'r2': r2}


def setup_distributed():
    """Initialise la distribution si torchrun est utilisé."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    """Main training and evaluation function."""
    
    is_dist, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # WandB uniquement sur rank 0
    if is_main_process():
        wandb.init(project='isbi2025', name=EXPERIMENT_NAME)
        (Path(OUTPUT_DIR) / EXPERIMENT_NAME).mkdir(exist_ok=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Set up data transforms
    transform_train = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(224, antialias=True),
            transforms.RandomRotation(degrees=(-10, 10)),
        ]),
        'image_only': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    transform_eval = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # Load metadata and data
    data_df = pd.read_csv(DATA_FILE)
    
    # Create datasets
    train_dataset = OrionImageDataset(
        data_df=data_df,
        transform=transform_train,
        all_biomarkers=CHANNELS,
        subset=DATASET_SPLITS['train']
    )

    val_dataset = OrionImageDataset(
        data_df=data_df,
        transform=transform_eval,
        all_biomarkers=CHANNELS,
        subset=DATASET_SPLITS['val'],
        is_test=True
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_dist else None

    # Create data loaders
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 4,
                            shuffle=False, sampler=val_sampler,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                            collate_fn=collate_fn)

    # Set up model and training
    model = get_model(num_outputs=len(CHANNELS)).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = masked_mse_loss

    # Training loop
    step = 0
    best_val_score = 0
    steps_since_best_val_score = 0

    while True:
        if is_dist:
            train_sampler.set_epoch(step)

        model.train()
        for inputs, labels, mask, _, _, _ in tqdm(train_loader, total=len(train_loader)):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels, mask)

            if torch.isnan(loss):
                continue
            
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping batch")
                continue
                
            loss.backward()
            optimizer.step()

            if is_main_process() and step % 100 == 0:
                wandb.log({'train_loss': loss.item()})

            # Validation
            if (step % EVAL_INTERVAL == 0) and (step > 0) and is_main_process():
                val_metrics = evaluate( # val_r2, val_ssim
                    model, val_loader, device, 
                    os.path.join(OUTPUT_DIR, EXPERIMENT_NAME), 
                    step, CHANNELS
                )
                
                #val_score = val_r2 + val_ssim
                val_score = (val_metrics["pearson_r"] + val_metrics["spearman_r"]) / 2
                wandb.log({
                    'val_pearson_r': val_metrics['pearson_r'],
                    'val_spearman_r': val_metrics['spearman_r'],
                    'val_r2': val_metrics['r2'],
                    'val_score': val_score
                })

                # Save best model
                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, EXPERIMENT_NAME, 'best_model.pth'))
                    steps_since_best_val_score = 0
                else:
                    steps_since_best_val_score += EVAL_INTERVAL

                scheduler.step(val_score)

                # Early stopping check
                if steps_since_best_val_score >= PATIENCE:
                    print(f'Early stopping after {step} steps')
                    return

            step += 1

    cleanup_distributed()

if __name__ == '__main__':
    main()
