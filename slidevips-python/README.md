# SlideVips

<p align="center">
  <img src="slidevips_logo.svg" alt="SlideVips Logo" style="height:250px;">
</p>

**SlideVips** is an efficient Python package for reading, processing, and managing whole slide images (WSI). Built on top of **pyvips**, SlideVips handles both H&E and high-dimensional multiplex immunofluorescence (mIF) images.

---

## ✨ Features

| Feature | SlideVips | OpenSlide |
|---|:---:|:---:|
| H&E WSI reading | ✅ | ✅ |
| Multi-channel mIF reading | ✅ | ❌ |
| Parallel tile reading/writing | ✅ | ❌ |
| PyTorch Dataset integration | ✅ | ❌ |
| Memory-efficient (pyvips backend) | ✅ | ❌ |
| Otsu-based tissue tile selection | ✅ | ❌ |

### Core capabilities

- **Fast reading:** Efficiently process large whole slide images using the pyvips backend.
- **Multiplex support:** Load specific channels from mIF images (e.g., OME-TIFF).
- **Parallel I/O:** Read and write batches of tiles using multiprocessing.
- **PyTorch integration:** Drop-in `SlideDataset` for training and inference pipelines.
- **Tissue selection:** Otsu-based filtering to skip background tiles.

---

## 📦 Installation

**Option 1 — Conda (recommended):**

```bash
conda install -c conda-forge -y pyvips python=3.11
pip install -e .  # run inside the slidevips-python/ folder
```

**Option 2 — pip / uv:**

```bash
apt-get install -y libvips-dev --no-install-recommends
pip install pyvips==3.1.1
pip install -e .
```

**Optional — jemalloc (strongly recommended for multiprocessing):**

> When using `SlideDataset` with PyTorch `DataLoader` (multiple workers), the default Python memory allocator may cause memory leaks. Using `jemalloc` eliminates this issue.

```bash
sudo apt-get install libjemalloc2
# Then run your scripts with:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python your_script.py
```

---

## 🚀 Quick Start

```python
from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu

# Open slides
slide_he = SlideVips(slide_path_he, mode="HE")
slide_if = SlideVips(slide_path_if, mode="IF", channels=[0, 1, 2])

# Get tissue tile positions
thumbnail = slide_he.get_thumbnail((1000, 1000))
tile_positions, _ = get_locs_otsu(thumbnail, slide_he.dimensions, (512, 512))

# Read a batch of tiles
tiles = slide_if.read_regions(tile_positions, level=0, tile_size=(512, 512))
```

---

## 📖 API Examples

### Reading Slides

```python
from slidevips import SlideVips

# H&E slide
slide_he = SlideVips(slide_path_he, mode="HE")

# mIF slide — load only channels 0 and 1
slide_if = SlideVips(slide_path_if, mode="IF", channels=[0, 1])
```

### Tile Extraction

```python
# Single tile
tile = slide_if.read_region(position, level, (tile_size_x, tile_size_y))

# Batch of tiles (parallel)
tiles = slide_if.read_regions(positions, level, (tile_size_x, tile_size_y))
```

### Writing Tiles to Disk

```python
# Single tile
slide_if.write_region(folder, position, level, (tile_size_x, tile_size_y), img_format=".tif")

# Batch of tiles (parallel)
slide_if.write_regions(folder, positions, level, (tile_size_x, tile_size_y), img_format=".tif")
```

### PyTorch Dataset

```python
from slidevips.torch_datasets import SlideDataset
import pandas as pd

slide_dataframe = pd.DataFrame({
    "in_slide_name": ["slide1", "slide2"],
    "in_slide_path": ["path_to_slide1", "path_to_slide2"]
})

dataframe = pd.DataFrame({
    "in_slide_name": ["slide1", "slide1", "slide2"],
    "x": [100, 200, 300],
    "y": [100, 200, 300],
    "level": [0, 1, 0],
    "tile_size_x": [256, 256, 256],
    "tile_size_y": [256, 256, 256]
})

dataset = SlideDataset(
    slide_dataframe,
    dataframe,
    mode="IF",
    channel_idxs=[0, 1, 2],
    preprocessing_fn=my_preprocessing_function,
    spatial_augmentation=my_spatial_augmentation,
    color_augmentation=my_color_augmentation
)
```

> **Memory note:** Always use `LD_PRELOAD=.../libjemalloc.so.2` when running with multiple DataLoader workers to avoid memory leaks from the pyvips allocator.
