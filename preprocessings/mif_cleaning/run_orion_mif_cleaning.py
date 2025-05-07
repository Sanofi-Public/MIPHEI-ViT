import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run mif_cleaning.py with required arguments.")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to slide_dataframe.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    command = [
        "python", "mif_cleaning.py",
        "--slide_dataframe_path", args.slide_dataframe_path,
        "--output_dir", args.output_dir,
        "--channel_names", "Hoechst", "AF1", "CD31", "CD45", "CD68", "Blank", "CD4", "FOXP3",
                         "CD8a", "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-Cadherin",
                         "PD-1", "Ki-67", "Pan-CK", "SMA",
        "--af_channel_name", "AF1",
        "--lambdas_path", "lambda_settings/orion.json",
        "--artifact_channel_name", "Blank",
        "--artifact_treshold", "2000",
        "--drop_channel_names", "Blank"
    ]

    print("Running command:\n", " ".join(command))
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()