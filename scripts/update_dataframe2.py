from pathlib import Path
import pandas as pd


SLIDE_DATAFRAME_PATH = "/root/workdir/slide_dataframe.csv"
IF_DIR = "/root/workdir/orion_if"
NUCLEI_CSV_DIR = "/root/workdir/orion_csv_nuclei_pos"
DATAFRAME_FOLDER = Path("../data")

if __name__ == "__main__":
    slide_dataframe = pd.read_csv(SLIDE_DATAFRAME_PATH)
    slide_dataframe["targ_slide_path"] = slide_dataframe["targ_slide_path"].apply(
        lambda x: str(Path(IF_DIR) / Path(x).name))
    slide_dataframe["nuclei_csv_path"] = slide_dataframe["nuclei_csv_path"].apply(
        lambda x: str(Path(NUCLEI_CSV_DIR) / Path(x).name))
    out_slide_dataframe_path = str(DATAFRAME_FOLDER / "slide_dataframe.csv")
    slide_dataframe.to_csv(out_slide_dataframe_path, index=False)

    train_dataframe_path = str(DATAFRAME_FOLDER / 'train_dataframe.csv')
    train_dataframe = pd.read_csv(train_dataframe_path)
    columns_rename = [name for name in train_dataframe.columns if "_pred" in name]
    columns_rename_dict = {name: name.replace("_pred", "_pos") for name in columns_rename}
    train_dataframe = train_dataframe.rename(columns=columns_rename_dict)
    train_dataframe.to_csv(train_dataframe_path, index=False)
    del train_dataframe

    val_dataframe_path = str(DATAFRAME_FOLDER / 'val_dataframe.csv')
    val_dataframe = pd.read_csv(val_dataframe_path)
    val_dataframe = val_dataframe.rename(columns=columns_rename_dict)
    val_dataframe.to_csv(val_dataframe_path, index=False)
    del val_dataframe

    test_dataframe_path = str(DATAFRAME_FOLDER / 'test_dataframe.csv')
    test_dataframe = pd.read_csv(test_dataframe_path)
    test_dataframe = test_dataframe.rename(columns=columns_rename_dict)
    test_dataframe.to_csv(test_dataframe_path, index=False)
    del test_dataframe
