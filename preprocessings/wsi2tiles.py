import pandas as pd
from slidevips import SlideVips
from pathlib import Path
from tqdm import tqdm
import gc

import concurrent.futures
from typing import List, Tuple


class SlideProcessor:
    def __init__(self, slide, scale_factor=None, interpolation=None):
        """
        Initialize the SlideProcessor with a slide.
        
        Args:
            slide: The slide object to process.
        """
        self.slide = slide
        self.scale_factor = scale_factor
        if self.scale_factor is not None and interpolation is None:
            raise ValueError("Interpolation must be provided if scale_factor is set.")
        
        if interpolation is not None:
            if interpolation not in ["linear", "nearest"]:
                raise ValueError("Interpolation must be 'linear' or 'nearest'.")
        self.interpolation = interpolation

    def apply_fn(self, folder, location: List[Tuple[int, int]], level: int, size: int, img_format=".png"):
        tile = self.slide._pyvips_crop_region(location, level, size)
        if self.scale_factor is not None:
            tile = tile.resize(self.scale_factor, kernel=self.interpolation)
        image_path = Path(folder) / "{}_{}_{}_{}_{}_{}{}".format(
            self.slide.slide_name, *location, level, *size, img_format)
        tile.write_to_file(image_path)
        return image_path
    
    def write_regions(self, folder: str, locations: List[Tuple[int, int]], level: int,
                      size: int, img_format: str = ".png") -> List[str]:
        """
        Write regions of an image to files in the specified folder in parallel
        using ThreadPoolExecutor. Uses a custom function instead of self.write_region.

        Args:
            folder (str): The folder path where the image regions will be saved.
            locations (list): A list of locations specifying the regions to be written.
            level (int): The zoom level of the image.
            size (int): The size of the image regions.
            img_format (str, optional): The format of the saved image files. Defaults to ".png".
            apply_fn (Callable, optional): Function to apply instead of self.write_region.

        Returns:
            list: A list of paths to the saved image files.
        """
        
        image_paths = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each location
            future_to_location = {executor.submit(
                self.apply_fn, folder, location,
                level, size, img_format): location for location in locations}

            # Collect the results as they become available
            for future in tqdm(concurrent.futures.as_completed(future_to_location), total=len(locations),
                               desc=f"Writing regions: {self.slide.slide_name}", leave=False):
                location = future_to_location[future]
                try:
                    image_path = future.result()
                    image_paths.append(image_path)
                except Exception as e:
                    print(f"Error occurred while writing region at location {location}: {e}")
        return image_paths
    
    def __del__(self):
        """
        Destructor method that closes the reader.
        """
        self.slide.close()


def wsi2tiles(slide_dataframe, dataframe, output_dir, mpp_target=None):

    Path(output_dir).mkdir(exist_ok=True)


    he_paths = []
    if_paths = []
    nuclei_paths = []
    for _, row in tqdm(slide_dataframe.iterrows(), total=len(slide_dataframe)):
        slide_name = row["in_slide_name"]
        dataframe_slide = dataframe[dataframe["in_slide_name"] == slide_name]
        if len(dataframe_slide) == 0:
            continue
        tile_positions = dataframe_slide[["x", "y"]].values
        assert dataframe_slide["level"].nunique() == 1
        assert dataframe_slide["tile_size_x"].nunique() == 1
        assert dataframe_slide["tile_size_y"].nunique() == 1
        level = dataframe_slide["level"].iloc[0]
        tile_size_x = dataframe_slide["tile_size_x"].iloc[0]
        tile_size_y = dataframe_slide["tile_size_y"].iloc[0]

        ## Convert H&E to tiles
        slide_he = SlideVips(row["in_slide_path"])
        mpp = slide_he.mpp
        if mpp_target is None:
            scale_factor = None
        else:
            scale_factor = mpp / mpp_target
        slide_processor = SlideProcessor(slide_he, scale_factor, "linear")
        he_dir = str(Path(output_dir) / "he")
        Path(he_dir).mkdir(exist_ok=True)
        he_paths_slide = slide_processor.write_regions(he_dir, tile_positions,
                            level, (tile_size_x, tile_size_y), img_format=".jpeg")
        slide_he.close()
        del slide_he, slide_processor
        he_paths.extend(he_paths_slide)
        gc.collect()

        ## Convert IF to tiles
        slide_if = SlideVips(row["targ_slide_path"], mode="IF")
        if slide_if.mpp != mpp:
            raise ValueError("Mismatch in MPP between H&E and IF slides.")
        slide_processor = SlideProcessor(slide_if, scale_factor, "linear")
        if_dir = str(Path(output_dir) / "if")
        Path(if_dir).mkdir(exist_ok=True)
        if_paths_slide = slide_processor.write_regions(if_dir, tile_positions,
                            level, (tile_size_x, tile_size_y), img_format=".tiff")
        slide_if.close()
        del slide_if, slide_processor
        if_paths.extend(if_paths_slide)
        gc.collect()

        ## Convert nuclei to tiles
        slide_nuclei = SlideVips(row["nuclei_slide_path"], mode="IF")
        if slide_nuclei.mpp != mpp:
            raise ValueError("Mismatch in MPP between H&E and IF slides.")
        slide_processor = SlideProcessor(slide_nuclei, scale_factor, "nearest")
        if_dir = str(Path(output_dir) / "nuclei")
        Path(if_dir).mkdir(exist_ok=True)
        nuclei_paths_slide = slide_processor.write_regions(if_dir, tile_positions,
                            level, (tile_size_x, tile_size_y), img_format=".tiff")
        slide_nuclei.close()
        del slide_nuclei, slide_processor
        nuclei_paths.extend(nuclei_paths_slide)
        gc.collect()

    dataframe_tiles = dataframe.copy()
    dataframe_tiles["image_path"] = he_paths
    dataframe_tiles["target_path"] = if_paths
    dataframe_tiles["nuclei_path"] = nuclei_paths
    return dataframe_tiles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply cleaning to WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide dataframe")
    parser.add_argument("--dataframe_path", type=str, required=True, help="Path to the input tile dataframe")
    parser.add_argument("--output_dataframe_path", type=str, required=True, help="Path to the output tile dataframe")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output tile directory")
    parser.add_argument("--mpp_target", type=float, default=None, help="Resolution of the output tiles in microns per pixel")
    
    args = parser.parse_args()

    slide_dataframe = pd.read_csv(args.slide_dataframe_path)
    dataframe = pd.read_csv(args.dataframe_path)
    output_dir = args.output_dir
    mpp_target = args.mpp_target
    output_dataframe_path = args.output_dataframe_path
    
    dataframe_tiles = wsi2tiles(slide_dataframe, dataframe, output_dir, mpp_target)
    dataframe_tiles.to_csv(output_dataframe_path, index=False)
