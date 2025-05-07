from pathlib import Path
import pandas as pd

from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu


TILE_SIZE = 512
TILE_OVERLAP = 0
LEVEL = 0
DATA_DIR = Path("../cloud-data/dds-efs/roger/orion")
SLIDE_DATAFRAME_PATH = "../data/slide_dataframe.csv"
DATAFRAME_FOLDER_SAVE = Path("../data")

if __name__ == "__main__":
    dirs_paths = [path for path in DATA_DIR.glob("*")]
    rows = []
    for dirs_path in dirs_paths:
        he_path = str(list(dirs_path.glob("*registered.ome.tif"))[0])
        he_name = Path(he_path).stem
        if_path = str(list(dirs_path.glob("*zlib.ome.tiff"))[0])
        assert he_path
        assert if_path
        rows.append([he_name, he_path, if_path])
    print(len(rows))

    slide_dataframe = pd.DataFrame(rows,
                                   columns=["in_slide_name", "in_slide_path", "targ_slide_path"])
    slide_dataframe.to_csv(SLIDE_DATAFRAME_PATH, index=False)

    slide_names = []
    xs = []
    ys = []

    for _, row in slide_dataframe.iterrows():
        slide_name, he_path = row["in_slide_name"], row["in_slide_path"]
        slide_he = SlideVips(he_path)

        thumbnail = slide_he.get_thumbnail((3000, 3000))
        tile_positions, tissue_percentages = get_locs_otsu(
            thumbnail, slide_he.level_dimensions[LEVEL], TILE_SIZE, TILE_OVERLAP, 0.07)

        slide_he.close()
        slide_names += [slide_name] * len(tile_positions)
        xs += tile_positions[..., 0].tolist()
        ys += tile_positions[..., 1].tolist()

    dataframe = pd.DataFrame(
        columns=["in_slide_name", "x", "y", "level", "tile_size_x", "tile_size_y"])
    dataframe["in_slide_name"] = slide_names
    dataframe["x"] = xs
    dataframe["y"] = ys
    dataframe["level"] = LEVEL
    dataframe["tile_size_x"] = TILE_SIZE
    dataframe["tile_size_y"] = TILE_SIZE

    dataframe.to_csv(str(DATAFRAME_FOLDER_SAVE / "dataframe.csv"), index=False)

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
