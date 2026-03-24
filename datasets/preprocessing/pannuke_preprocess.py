"""
Pannuke preprocessing:
- Build image/inst/sem dataframe
- Extract nuclei-level cell types
- Save nuclei cell type table (CSV + Parquet)
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import trange, tqdm
from skimage.measure import label


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def connected_components_per_label(instances: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """Reassign unique instance IDs for each connected region within each integer label."""
    # # https://github.com/Mr-TalhaIlyas/Prerpcessing-PanNuke-Nuclei-Instance-Segmentation-Dataset/blob/master/scripts/process_pannuke_std.py
    new_mask = np.zeros_like(instances, dtype=np.int32)
    next_id = 1

    for lbl in np.unique(instances):
        if lbl == 0:
            continue  # skip background
        mask = (instances == lbl)
        cc = label(mask, connectivity=connectivity)
        cc[cc > 0] += next_id - 1
        new_mask += cc
        next_id = new_mask.max() + 1

    return new_mask


def convert_fold_npy_to_pngs(folds, data_dir, output_dir):
    # https://github.com/Mr-TalhaIlyas/Prerpcessing-PanNuke-Nuclei-Instance-Segmentation-Dataset/blob/master/scripts/process_pannuke_std.py

    if not data_dir.endswith('/'):
        data_dir += '/'
    if not output_dir.endswith('/'):
        output_dir += '/'

    for i, j in enumerate(folds):
    
        # get paths
        print('Loading Data for {}, Wait...'.format(j))
        img_path =data_dir + j + '/images/fold{}/images.npy'.format(i+1)
        type_path = data_dir + j + '/images/fold{}/types.npy'.format(i+1)
        mask_path = data_dir + j + '/masks/fold{}/masks.npy'.format(i+1)
        print(40*'=', '\n', j, 'Start\n', 40*'=')
        
        # laod numpy files
        masks = np.load(file=mask_path, mmap_mode='r') # read_only mode
        images = np.load(file=img_path, mmap_mode='r') # read_only mode
        types = np.load(file=type_path) 
        
        # creat directories to save images
        try:
            os.mkdir(output_dir + j)
            os.mkdir(output_dir + j + '/images')
            os.mkdir(output_dir + j + '/sem_masks')
            os.mkdir(output_dir + j + '/inst_masks')
        except FileExistsError:
            pass
            
        
        for k in trange(images.shape[0], desc='Writing files for {}'.format(j), total=images.shape[0]):
            
            raw_image =  images[k,:,:,:].astype(np.uint8)
            raw_mask = masks[k,:,:,:]
            sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
            # swaping channels 0 and 5 so that BG is at 0th channel
            sem_mask = np.where(sem_mask == 5, 6, sem_mask)
            sem_mask = np.where(sem_mask == 0, 5, sem_mask)
            sem_mask = np.where(sem_mask == 6, 0, sem_mask)

            tissue_type = types[k]
            instances = connected_components_per_label(np.int32(raw_mask[..., :-1].max(axis=-1)))
            instances = np.uint16(instances)
            
            # # for plotting it'll slow down the process considerabelly
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(instances)
            # ax[1].imshow(sem_mask)
            # ax[2].imshow(raw_image)
            
            # save file in op dir
            Image.fromarray(sem_mask).save(output_dir + '/{}/sem_masks/sem_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
            Image.fromarray(instances).save(output_dir +'/{}/inst_masks/inst_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
            Image.fromarray(raw_image).save(output_dir +'/{}/images/img_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 


def find_image_paths(base_dir: Path):
    """Collect all PNG image paths from Fold 1/2/3."""
    folds = [base_dir / "Fold 1" / "images",
             base_dir / "Fold 2" / "images",
             base_dir / "Fold 3" / "images"]

    paths = []
    for fold in folds:
        paths.extend(fold.glob("*.png"))

    return [str(p) for p in paths]


def build_slide_dataframe(image_paths):
    """Build main image dataframe."""
    df = pd.DataFrame()
    df["in_slide_name"] = [Path(p).stem for p in image_paths]
    df["image_path"] = image_paths

    df["nuclei_path"] = df["image_path"].apply(
        lambda p: p.replace("/images/", "/inst_masks/")
                  .replace("/img", "/inst")
    )
    df["sem_masks"] = df["image_path"].apply(
        lambda p: p.replace("/images/", "/sem_masks/")
                  .replace("/img", "/sem")
    )

    df["organ"] = df["in_slide_name"].apply(lambda x: x.split("_")[1])

    return df


def validate_paths(df):
    """Ensure all referenced paths exist."""
    assert df["image_path"].apply(lambda x: Path(x).exists()).all(), "Missing images"
    assert df["nuclei_path"].apply(lambda x: Path(x).exists()).all(), "Missing instance masks"
    assert df["sem_masks"].apply(lambda x: Path(x).exists()).all(), "Missing semantic masks"


def extract_cell_types(df):
    """Extract per-nucleus cell types from instance + semantic masks."""
    records = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing nuclei"):
        slide = row["in_slide_name"]

        inst = np.array(Image.open(row["nuclei_path"]), dtype=np.int32)
        sem = np.array(Image.open(row["sem_masks"]), dtype=np.int32)

        labels = np.unique(inst)
        labels = labels[labels != 0]

        for lbl in labels:
            mask = inst == lbl

            vals_ct, counts_ct = np.unique(sem[mask], return_counts=True)
            major_idx = counts_ct.argmax()
            major_class = int(vals_ct[major_idx])
            frac_major = counts_ct[major_idx] / counts_ct.sum()

            records.append({
                "in_slide_name": slide,
                "cell_id": int(lbl),
                "cell_class": major_class,
            })

    return pd.DataFrame(records)


def one_hot_encode_cell_types(nuclei_df, nuclei_classes):
    """Convert numeric classes → names → one-hot."""
    nuclei_df = nuclei_df.rename(columns={"in_slide_name": "slide_name",
                                          "cell_id": "label"})
    nuclei_df["slide_name"] = nuclei_df["slide_name"].astype("category")

    min_class = nuclei_df["cell_class"].min()
    nuclei_df["ct_name"] = nuclei_df["cell_class"].map(
        lambda x: nuclei_classes[x - min_class]
    )

    one_hot = pd.get_dummies(nuclei_df["ct_name"])
    nuclei_df = pd.concat([nuclei_df, one_hot], axis=1)
    nuclei_df["ct_name"] = nuclei_df["ct_name"].astype("category")

    return nuclei_df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(pan_path: Path):
    pan_path = pan_path.resolve()
    print(f"Using PanNuke base directory: {pan_path}")

    # ---- Step 1: Convert NPY to PNGs
    orig_data_dir = pan_path / "orig_data/"
    process_data_dir = pan_path / "process_data/"
    process_data_dir.mkdir(exist_ok=True)
    folds = [p.name for p in orig_data_dir.iterdir() if p.is_dir()]
    convert_fold_npy_to_pngs(folds, str(orig_data_dir), str(process_data_dir))
    print("Converted NPY files to PNGs.")

    # ---- Step 2: Collect images
    image_paths = find_image_paths(pan_path / "process_data")
    print(f"Found {len(image_paths)} images.")

    # ---- Step 3: Build dataframe
    df = build_slide_dataframe(image_paths)
    validate_paths(df)

    df_out = pan_path / "pannuke_dataframe.csv"
    df.to_csv(df_out, index=False)
    print(f"Saved dataframe: {df_out}")

    # ---- Step 4: Nuclei extraction
    nuclei_df = extract_cell_types(df)

    # ---- Step 5: One-hot + Parquet
    nuclei_classes = [
        "Neoplastic cells",
        "Inflammatory",
        "Connective/Soft tissue cells",
        "Dead Cells",
        "Epithelial",
        "Background"
    ]

    nuclei_df = one_hot_encode_cell_types(nuclei_df, nuclei_classes)

    nuclei_parquet = pan_path / "nuclei_dataframe.parquet"
    nuclei_df.to_parquet(nuclei_parquet, compression=None)

    print(f"Saved nuclei types Parquet: {nuclei_parquet}")
    print("Done! ✨")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanNuke preprocessing.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to `data/pannuke` folder (the folder containing process_data/)."
    )
    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir) / "pannuke")
    main(Path(args.data_dir))
