# SlideVips

## Overview
**SlideVips** is an efficient and powerful Python package designed for reading, processing, and managing whole slide images (WSI). Built on top of the fast **pyvips** library, SlideVips offers memory-efficient operations and integration with **PyTorch** for creating datasets. This makes it an excellent tool for researchers and developers working with multiplex immunofluorescence (mIF) or H&E-stained images in digital pathology.

## Features

### Core Capabilities
- **Fast Reading:** Read and process whole slide images efficiently, even for large datasets.
- **Multiplex Support:** Handle multiplex immunofluorescence (mIF) images and H&E images with ease.
- **Multi-Channel Reading:** Load specific channels of mIF images as needed.
- **Multi-Process Operations:** Perform batch tile reading and writing using multiprocessing for enhanced speed.
- **PyTorch Dataset Integration:** Easily create datasets for machine learning workflows.

### Example Functionality

#### 1. Reading Whole Slide Images
```python
from slidevips import SlideVips

# Reading a multiplex immunofluorescence (mIF) slide
slide_if = SlideVips(slide_path_if, mode="IF", channels=[0, 1])  # Read channel 0 and 1

# Reading an H&E slide
slide_he = SlideVips(slide_path_he, mode="HE")
```

#### 2. Extracting Thumbnails and Tile Positions
```python
from slidevips.tiling import get_locs_otsu

# Extract a thumbnail
thumbnail_he = slide_he.get_thumbnail((1000, 1000))

# Perform Otsu tile selection and get tile positions at level 0
tile_positions, _ = get_locs_otsu(thumbnail_he, slide_he.dimensions, (512, 512))
```

#### 3. Tile Extraction
```python
# Extract a single tile
tile_if = slide_if.read_region(tile_position, level, (tile_size_x, tile_size_y))

# Extract multiple tiles in parallel
tiles_if = slide_if.read_regions(tile_positions, level, (tile_size_x, tile_size_y))
```

#### 4. Writing Tiles to Disk
```python
# Write a single tile to a specified folder
slide_if.write_region(folder, tile_position, level, (tile_size_x, tile_size_y), img_format=".tif")

# Write multiple tiles to disk in parallel
slide_if.write_regions(folder, tile_positions, level, (tile_size_x, tile_size_y), img_format=".tif")
```

#### 5. Creating PyTorch Datasets
```python
from slidevips.torch_datasets import SlideDataset
import pandas as pd

# Define slide metadata
slide_dataframe = pd.DataFrame({
    "in_slide_name": ["slide1", "slide2"],
    "in_slide_path": ["path_to_slide1", "path_to_slide2"]
})

# Define tile metadata
dataframe = pd.DataFrame({
    "in_slide_name": ["slide1", "slide1", "slide2"],
    "x": [100, 200, 300],
    "y": [100, 200, 300],
    "level": [0, 1, 0],
    "tile_size_x": [256, 256, 256],
    "tile_size_y": [256, 256, 256]
})

# Create a PyTorch dataset
dataset = SlideDataset(
    slide_dataframe,
    dataframe,
    mode="IF",
    channel_idxs=[0, 1, 2],
    preprocessing_fn=my_preprocessing_function,  # E.g., normalization
    spatial_augmentation=my_spatial_augmentation,  # Albumentations augmentations
    color_augmentation=my_color_augmentation  # Color-specific augmentations
)
```

## Installation
To install SlideVips, follow these steps:

1. Install **pyvips** and Python 3.11 via Conda:
   ```bash
   conda install -c conda-forge -y pyvips python=3.11
   ```
2. Install the SlideVips package:
   ```bash
   pip install -e .  # Run this inside the package folder
   ```
