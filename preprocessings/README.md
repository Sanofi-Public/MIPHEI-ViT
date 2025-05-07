# Preprocessings

This folder contains all preprocessing steps used to generate cleaned, tile-based datasets from spatially aligned raw multiplex immunofluorescence (mIF) and H&E slides. The pipeline includes:

- WSI miF cleaning
- WSI nuclei segmentation using CellPose (Dapi) and HoverFast (H&E)
- Artifact detection and removal
- Single-cell extraction and gating for cell type identification
- Image tiling and normalization statistics extraction
- Generation of final tile datasets ready for training

> ⚠️ The preprocessing scripts rely heavily on the slidevips package we developed URL.
> 
> 
> 💡 We recommend running the full pipeline on a machine with at least **64 GB of RAM**.
> 

This folder includes scripts to reproduce the preprocessing pipelines for both the **ORION** and **HEMIT** datasets.

**Note:** Preprocessing steps can be **time-consuming**, as they are not fully optimized and operate on large-scale whole slide image (WSI) data.

## 📥 Reproduce Dataset preprocessings

### Hemit:

To reproduce the HEMIT dataset preprocessing, run:

```bash
hemit_preprocessings.ipynb
```

### ORION:

To regenerate the ORION preprocessed dataset:

1. Download the original ORION raw data by following the instructions [here on Zenodo](https://zenodo.org/records/7637988).
2. Prepare a slide dataframe (`slide_dataframe`) with the following columns:
    - `in_slide_name`: a unique ID (e.g., the slide name)
    - `in_slide_path`: the path to the corresponding H&E image
    - `targ_slide_path`: the path to the aligned mIF image
3. Update the `--slide_dataframe_path` and other paths in the commands below to reflect your local directory structure.

## ▶️ Running the Preprocessing Pipeline

Run the following steps in order to reproduce results on ORION dataset.

### 1. Orion mIF Cleaning

Use the Napari-based tool in `mif_cleaning/napari_af_subtraction_tool.py`  to manually select optimal λ (lambda) and β (beta) values for autofluorescence subtraction. Alternatively, you can use our predefined hyperparameters located in: `mif_cleaning/lambda_settings/orion.json`

Then run the following script to remove autofluorescence and normalize the mIF images to 8-bit:

```bash
python run_orion_mif_cleaning.py --slide_dataframe_path <path_to_slide_dataframe.csv> --output_dir <path_to_output_clean_folder>
```

This script also generates a `clean_slide_dataframe.csv` that maps original `if_path` values to the corresponding cleaned WSI paths.

### 2. Nuclei Segmentation

This step runs Cellpose inference across the full WSI using the DAPI channel and applies nuclei expansion. It generates WSI TIFF label images, where:

- Channel 1 contains nuclei instance labels
- Channel 2 contains a boundary map for visualization

You can download the CellPose model here.

```bash
python run_orion_cellpose_segmentation.py \\
    --slide_dataframe_path <path_to_clean_slide_dataframe.csv> \\
    --cellpose_ckpt_path <path_to_dapi_checkpoint.ckpt> \\
    --output_dir <path_to_output_nuclei_folder>
```

### 3. Single-Cell Analysis

This step computes mean mIF expression per nucleus. Afterward, use the provided notebook to perform GMM clustering and define pseudo-cell types.

```bash
python extract_expression_matrix.py \
--slide_dataframe_path <path_to_clean_slide_dataframe.csv> \
--output_dir <path_to_output_single_cell_csv>

# Then manually run:
artifact_removal.ipynb
```

### 4. General Tiling

Run the tiling script to generate a dataframe containing tile positions and metadata for tiles within the H&E tissue region.

```bash
python tiling.py \
--slide_dataframe_path <path_to_clean_slide_dataframe.csv> \
--output_path <output_tile_dataframe.csv> \
--level 0 \
--tile_size 512
```

### 4. Artifact removal

Run the following scripts to remove H&E and mIF artifacts. Run artifact_removal.ipynb to apply filtering and create a new tile dataframe without tile artifacts.

```bash
python extract_embeddings.py \
--slide_dataframe_path <path_to_clean_slide_dataframe.csv> \
--dataframe_path <path_to_tile_dataframe.csv> \
--output_path <path_to_tile_embeddings.npy> \
--downsample_2x

python orion_extract_if_artifact_props.py \
--slide_dataframe_path <path_to_clean_slide_dataframe.csv> \
--dataframe_path <path_to_tile_dataframe.csv> \
--output_path <path_to_artifact_props.npy>

Then manually run:

artifact_removal.ipynb
```

### 5. Final Tile Dataset Creation

The following script extracts tiles from WSI images into individual tile files — JPEG for H&E, and TIFF for mIF and nuclei masks.

The second script computes per-channel statistics (mean and standard deviation) across all extracted tiles, for H&E and mIF channels.

```bash
python wsi2tiles.py \
--slide_dataframe_path <path_to_clean_slide_dataframe.csv> \
--dataframe_path <path_to_clean_tile_dataframe.csv> \
--output_dataframe_path <output_tile_dataframe.csv> \
--output_dir <tile_output_folder> \
--mpp_target 0.5

python get_mean_std_channels.py \
--dataframe_path <path_to_tile_dataframe.csv> \
--output_path <output_channel_stats.json> \
--channel_names Hoechst CD31 CD45 CD68 CD4 FOXP3 CD8a CD45RO CD20 PD-L1 CD3e CD163 E-Cadherin PD-1 Ki-67 Pan-CK SMA
```

## 🧩 Additional Preprocessings

We also provide scripts for registration and H&E-based inference used in the **IMMUcan** project:

- **Registration**: Built on the [Valis](https://valis.readthedocs.io/) framework, our code enables automatic H&E-to-mIF registration. It includes tools to identify and exclude poorly aligned pairs.
- **H&E Segmentation**: We offer a modified version of **HoverFast** that is compatible with `slidevips`.

## 🧪 Preprocessing Outputs

Running all preprocessing steps will generate the following artifacts:

- 📋 **Cleaned Slide Metadata**
    - A table summarizing included slides with valid annotations and quality control.
- 🧼 **Cleaned Tile Metadata**
    - A filtered DataFrame of tiles passing artifact removal (H&E and mIF).
- 🖼️ **Full WSI Clean mIF Images**
    - Denoised, autofluorescence-corrected multi-channel TIFFs (`uint8`), one per slide.
- 🧬 **Nuclei Segmentations**
    - Label and boundary masks as full-resolution TIFFs for each WSI.
- 📊 **Single-Cell Expression Matrices**
    - `.csv` files with cell-level marker expression, spatial location, and inferred cell types.
- 🧩 **Final Image Tiles**
    - Patch-level crops extracted from H&E, mIF, and segmentation masks for model training/evaluation.
- 📈 **Per-Channel Normalization Stats**
    - Mean and standard deviation for each mIF channel, used for normalization across tiles/slides.

## Citations

This work rely a lot on this two **open-source datasets**:

- **OrionCRC**
    
    Source: [labsyspharm/ORION-CRC – Zenodo](https://zenodo.org/records/7637988)
    
    Registered, co-stained whole-slide images (WSIs) of colorectal cancer tissue acquired via mIF and H&E restaining on the same slide.
    
    **Citation**:
    
    Lin J. *labsyspharm/ORION-CRC* [dataset]. Zenodo. 2023. doi:10.5281/zenodo.7637988
    
- **HEMIT**
    
    Source: [Mendeley Data](https://data.mendeley.com/datasets/3gx53zm49d/1)
    
    Multi-institutional H&E and mIF dataset with consecutive section alignment from formalin-fixed paraffin-embedded (FFPE) tissue blocks.
    
    **Citation**:
    
    Bian C, Philips B, Cootes T, et al. *HEMIT: H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator*. arXiv preprint arXiv:2403.18501, 2024.
    

We also used data from the **IMMUcan** dataset for validation purposes; however, it is not redistributed here as the dataset remains private.
