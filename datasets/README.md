## 📦 Download and Preprocess the Datasets

Use the following script to automatically download and preprocess the datasets:

```bash
./data_setup.sh --data_dir <path> --all
```

You can also select specific datasets:

```bash
./data_setup.sh --data_dir <path> --dataset orion --dataset hemit
```

**Note:**

When downloading **OrionCRC**, you may occasionally encounter a “bit / packet” transfer error. If this happens, simply re-run the download — it will resume automatically.

Once the data is downloaded and preprocessed, update the corresponding config files in the `config/data/` folder so they point to the correct paths.

Make sure the required packages are installed beforehand.

---

### 🔧 System dependencies

Install the required system tools:

```bash
sudo apt-get install -y aria2 unzip p7zip-full
```

### 📚 Python dependencies

Install the repository's Python dependencies:

```bash
pip install -r requirements.txt
pip install kaggle huggingface_hub
```

If you use Lizard or PathoCell, authenticate once:

```bash
# Kaggle (Lizard)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/   # you can download this file from your Kaggle account settings
chmod 600 ~/.kaggle/kaggle.json

# HuggingFace (PathoCell)
huggingface-cli login

```

---

### 📁 Dataset Structure

Each dataset configuration (in `config/data/`) must follow the structure below.

This format applies to **OrionCRC**, **HEMIT**, **PathoCell, Lizard, PanNuke**, and any **custom dataset** you want to plug in.

```
WSI datasets:  # if inputs/targets are in ome.tiff/ndpi etc.
    slide_dataframe:
        - in_slide_name        # unique slide ID
        - in_slide_path        # path to H&E WSI
        - targ_slide_path      # path to mIF WSI (optional)
        - nuclei_path             # path to nuclei instance WSI

    train/val/test:
        - in_slide_name        # matches slide ID slide_dataframe
        - x, y                 # tile coordinates at level 0
        - level                # WSI pyramid level
        - tile_size_x          # tile width in pyramid level
        - tile_size_y          # tile height in pyramid level

Tile datasets:  # if tiles were extracted as .PNG, .tiff files
    train/val/test:
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

If you want to add a new dataset, place your data in its own folder, create the corresponding config file in `config/data/`, and implement a matching dataset evaluator in `evaluators/dataset_evaluators/`.
