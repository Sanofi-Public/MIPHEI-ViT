import pandas as pd
from slidevips import SlideVips
from slidevips.tiling import get_locs_otsu

from tqdm import tqdm


def tile_dataset(slide_dataframe, level, tile_size_lvl_0, output_dataframe_path):

    dataframe = []

    for _, row in tqdm(slide_dataframe.iterrows(), total=len(slide_dataframe)):
        slide_name = row["in_slide_name"]
        slide_path = row["in_slide_path"]
        slide_he = SlideVips(slide_path)
        tile_size = int(tile_size_lvl_0 / slide_he.level_downsamples[level])

        thumbnail = slide_he.get_thumbnail((3000, 3000))
        tile_positions, _ = get_locs_otsu(thumbnail, slide_he.dimensions, tile_size_lvl_0)

        dataframe_slide = pd.DataFrame(
            columns=["in_slide_name", "x", "y",
                    "level", "tile_size_x", "tile_size_y"])
        dataframe_slide["in_slide_name"] = [slide_name] * len(tile_positions)
        dataframe_slide["x"] = tile_positions[..., 0]
        dataframe_slide["y"] = tile_positions[..., 1]
        dataframe_slide["level"] = level
        dataframe_slide["tile_size_x"] = tile_size
        dataframe_slide["tile_size_y"] = tile_size
        dataframe.append(dataframe_slide)

        slide_he.close()
        del thumbnail

    dataframe = pd.concat(dataframe, ignore_index=True)
    dataframe.to_csv(output_dataframe_path, index=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply cleaning to WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where the cleaned slides will be saved")

    parser.add_argument("--level", type=int, required=True, help="Level in pyramid to select tiles")
    parser.add_argument("--tile_size", type=int, required=True, help="Tile size at lowest level in pyramid")
    args = parser.parse_args()

    slide_dataframe_path = args.slide_dataframe_path
    slide_dataframe = pd.read_csv(slide_dataframe_path)
    output_path = args.output_path
    level = args.level
    tile_size = args.tile_size

    tile_dataset(slide_dataframe, level, tile_size, output_path)
