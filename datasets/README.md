# 📦 Datasets

This folder provides download scripts and configuration guidelines for all datasets used in MIPHEI-ViT experiments.

Preprocessed versions of **OrionCRC** and **HEMIT** are publicly available on Zenodo:

🔗 **Zenodo archive:** https://doi.org/10.5281/zenodo.15340874

---

## 🗂️ Supported Datasets

| Dataset | Modality | Task | Source |
|---|---|---|---|
| **OrionCRC** | H&E + 16-marker mIF (Hoechst, CD31, CD45, CD68, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e, CD163, E-cadherin, Ki67, Pan-CK, SMA) | Pixel-level mIF prediction + cell-type classification | [Zenodo](https://doi.org/10.5281/zenodo.15340874) |
| **HEMIT** | H&E + 3-marker mIF (CD3, Pan-CK) | Pixel-level mIF prediction + cell-level evaluation | [HEMIT paper](https://github.com/BianChang/HEMIT-DATASET) |
| **PathoCell** | H&E + 56-marker mIF | Pixel-level mIF prediction + cell-type classification | HuggingFace |
| **Lizard** | H&E + human-annotated cell types | Cell-level evaluation only (no mIF) | Kaggle |
| **PanNuke** | H&E + human-annotated cell types | Cell-level evaluation only (no mIF) | warwick.ac.uk |

---

## ⚙️ Setup

### System dependencies

```bash
sudo apt-get install -y aria2 unzip p7zip-full
```

### Python dependencies

```bash
pip install -r requirements.txt
pip install kaggle huggingface_hub
```

If you use **Lizard** or **PathoCell**, authenticate once:

```bash
# Kaggle (Lizard)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/   # download from your Kaggle account settings
chmod 600 ~/.kaggle/kaggle.json

# HuggingFace (PathoCell)
hf auth login
```

---

## 📥 Download and Preprocess

Use the following script to automatically download and preprocess all datasets:

```bash
./setup_data.sh --data_dir <path> --all
```

Or select specific datasets:

```bash
./setup_data.sh --data_dir <path> --dataset orion --dataset hemit
```

> **Note:** When downloading **OrionCRC**, you may occasionally encounter a download error. Re-running the command will resume automatically.

After downloading, check that config files in `config/data/` point to the correct paths.

---

## 📁 Dataset Configuration Format

Each dataset configuration (in `config/data/`) must follow the structure below. This applies to **OrionCRC**, **HEMIT**, **PathoCell**, **Lizard**, **PanNuke**, and any custom dataset.

```
WSI datasets:  # inputs/targets are ome.tiff / ndpi / svs files
    slide_dataframe:
        - in_slide_name        # unique slide ID
        - in_slide_path        # path to H&E WSI
        - targ_slide_path      # path to mIF WSI (optional)
        - nuclei_path          # path to nuclei instance WSI

    train / val / test:
        - in_slide_name        # matches slide ID in slide_dataframe
        - x, y                 # tile coordinates at level 0
        - level                # WSI pyramid level
        - tile_size_x          # tile width in pyramid level
        - tile_size_y          # tile height in pyramid level

Tile datasets:  # inputs/targets are pre-extracted .PNG or .tiff files
    train / val / test:
        - image_path           # H&E tile
        - target_path          # mIF tile (optional)
        - nuclei_path          # nuclei instances tile

Common:
    nuclei_classes: [list]     # names of nuclei categories
    nuclei_dataframe: parquet with:
        - label                # nucleus ID (same in instance tiles/WSIs)
        - slide_name           # FoV name: WSI or Tile ID
        - one column per nuclei class (bool)

mIF only:
    marker_metadata_path       # CSV with "Marker Name" + "Index" matching mIF channels
    targ_channel_names: [list] # markers used during training
```

---

## ➕ Adding a Custom Dataset

1. Place your data in its own folder.
2. Create a config file in `config/data/` following the structure above.
3. Implement a matching dataset evaluator in `evaluators/dataset_evaluators/`.
