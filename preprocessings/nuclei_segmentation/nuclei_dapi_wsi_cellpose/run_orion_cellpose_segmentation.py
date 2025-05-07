import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run mif_cleaning.py with required arguments.")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to slide_dataframe.csv")
    parser.add_argument("--cellpose_ckpt_path", type=str, required=True, help="Path of the CellPose checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    command = [
        "python", "cellpose_wsi_inference.py",
        "--slide_dataframe_path", args.slide_dataframe_path,
        "--idx_dapi", "0",
        "--cellpose_ckpt_path", args.cellpose_ckpt_path,
        "--output_dir", args.output_dir,
    ]

    print("Running command:\n", " ".join(command))
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
