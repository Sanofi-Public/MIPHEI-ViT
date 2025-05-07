import os
import pandas as pd
from pathlib import Path
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, help='Slide dataframe path')
    parser.add_argument('--image_folder', type=str, help='Image folder')
    args = parser.parse_args()

    # Load the dataframe
    df = pd.read_csv(args.dataframe_path)  # Update with actual path

    # Define base folder where subfolders will be created
    base_folder = Path(args.image_folder)  # Update with your target base directory
    base_folder.mkdir(exist_ok=True)  # Ensure base folder exists

    for _, row in df.iterrows():
        subfolder = base_folder / row["in_slide_name"]
        subfolder.mkdir(exist_ok=True)  # Create subfolder

        # Define paths for symlinks
        in_slide_symlink = subfolder / Path(row["in_slide_path"]).name
        targ_slide_symlink = subfolder / Path(row["targ_slide_path"]).name

        # Create symbolic links if they don't already exist
        if not in_slide_symlink.exists():
            os.symlink(row["in_slide_path"], str(in_slide_symlink))

        if not targ_slide_symlink.exists():
            os.symlink(row["targ_slide_path"], str(targ_slide_symlink))

    print("Symlinks created successfully.")
