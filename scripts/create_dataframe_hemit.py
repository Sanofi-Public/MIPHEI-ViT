from pathlib import Path
import pandas as pd

from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu


DATA_DIR = Path("/root/workdir/HEMIT_dataset")
DATAFRAME_FOLDER_SAVE = Path("../data")

if __name__ == "__main__":
    train_dataframe = pd.DataFrame(columns=['in_slide_name', 'image_path', 'target_path', 'nuclei_path', 'nuclei_csv_path'])
    val_dataframe = pd.DataFrame(columns=['in_slide_name', 'image_path', 'target_path', 'nuclei_path', 'nuclei_csv_path'])
    test_dataframe = pd.DataFrame(columns=['in_slide_name', 'image_path', 'target_path', 'nuclei_path', 'nuclei_csv_path'])


    train_dataframe["image_path"] = [str(fn) for fn in (Path(DATA_DIR) / "train/input").glob("*.tif")]
    train_dataframe["in_slide_name"] = train_dataframe["image_path"].apply(lambda x: Path(x).name)
    train_dataframe["target_path"] = train_dataframe["image_path"].apply(lambda x: x.replace("/input/", "/label/"))
    train_dataframe["nuclei_path"] = train_dataframe["image_path"].apply(lambda x: x.replace("/input/", "/mask/"))

    val_dataframe["image_path"] = [str(fn) for fn in (Path(DATA_DIR) / "val/input").glob("*.tif")]
    val_dataframe["in_slide_name"] = val_dataframe["image_path"].apply(lambda x: Path(x).name)
    val_dataframe["target_path"] = val_dataframe["image_path"].apply(lambda x: x.replace("/input/", "/label/"))
    val_dataframe["nuclei_path"] = val_dataframe["image_path"].apply(lambda x: x.replace("/input/", "/mask/"))

    test_dataframe["image_path"] = [str(fn) for fn in (Path(DATA_DIR) / "test/input").glob("*.tif")]
    test_dataframe["in_slide_name"] = test_dataframe["image_path"].apply(lambda x: Path(x).name)
    test_dataframe["target_path"] = test_dataframe["image_path"].apply(lambda x: x.replace("/input/", "/label/"))
    test_dataframe["nuclei_path"] = test_dataframe["image_path"].apply(lambda x: x.replace("/input/", "/mask/"))


    print(len(train_dataframe), len(val_dataframe), len(test_dataframe))

    dataframe = pd.concat((train_dataframe, val_dataframe, test_dataframe))
    slide_dataframe = pd.DataFrame()
    slide_dataframe["in_slide_name"] = dataframe["in_slide_name"]
    slide_dataframe["nuclei_csv_path"] = dataframe["image_path"].apply(lambda x: x.replace("/input/", "/csv/").replace(".tif", ".csv"))

    assert dataframe["image_path"].apply(lambda x: Path(x).exists()).all()
    assert dataframe["target_path"].apply(lambda x: Path(x).exists()).all()
    assert dataframe["nuclei_path"].apply(lambda x: Path(x).exists()).all()

    slide_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "slide_dataframe_hemit.csv"), index=False)
    dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "dataframe_hemit.csv"), index=False)
    train_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "train_dataframe_hemit.csv"), index=False)
    val_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "val_dataframe_hemit.csv"), index=False)
    test_dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "test_dataframe_hemit.csv"), index=False)
