from pathlib import Path
import pandas as pd

DATAFRAME_CSV_PATH = "/root/workdir/data/dataframe.csv"
DATAFRAME_FOLDER_SAVE = Path("../data")

if __name__ == "__main__":

    dataframe = pd.read_csv(DATAFRAME_CSV_PATH)

    test_slides = ['19510_C11_US_SCAN_OR_001__151039-registered.ome',
                   '18459_LSP10364_US_SCAN_OR_001__092347-registered.ome']
    val_slides = ['19510_C19_US_SCAN_OR_001__153041-registered.ome',
                  '19510_C30_US_SCAN_OR_001__155702-registered.ome']
    train_dataframe = dataframe[~dataframe["in_slide_name"].isin(val_slides + test_slides)]
    val_dataframe = dataframe[dataframe["in_slide_name"].isin(val_slides)]
    test_dataframe = dataframe[dataframe["in_slide_name"].isin(test_slides)]

    print(len(train_dataframe), len(val_dataframe), len(test_dataframe))
    train_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "train_dataframe.csv"), index=False)
    val_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "val_dataframe.csv"), index=False)
    test_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "test_dataframe.csv"), index=False)
