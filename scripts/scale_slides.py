import pyvips
import pandas as pd
from slidevips import SlideVips
from pathlib import Path
import ome_types

from tqdm import tqdm

OUT_DIR = Path("/root/workdir/slides_scale")
SLIDE_DATAFRAME_PATH = "/root/workdir/data/slide_dataframe.csv"

if __name__ == "__main__":
    slide_dataframe = pd.read_csv(SLIDE_DATAFRAME_PATH)
    paths = slide_dataframe["in_slide_path"].to_list()

    for path in tqdm(paths):
        output_path = str(OUT_DIR / Path(path).name)
        slide = SlideVips(path)
        scale = slide.mpp / 0.245
        slide.resize(scale)

        final_img = slide.pyramid_image[0]
        final_img = final_img.copy()
        image_height = final_img.height  # one channel only
        ome_metadata = ome_types.from_tiff(path)
        ome_metadata.images[0].pixels.type = "uint8"
        ome_metadata.images[0].pixels.size_x = final_img.width
        ome_metadata.images[0].pixels.size_y = image_height
        ome_metadata.images[0].pixels.physical_size_x = 0.245
        ome_metadata.images[0].pixels.physical_size_y = 0.245

        ome_xml_metadata = ome_metadata.to_xml()

        final_img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
        final_img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

        final_img.tiffsave(output_path,
                        compression="jpeg",
                        predictor="none",
                        pyramid=True,
                        tile=True,
                        tile_width=512,
                        tile_height=512,
                        bigtiff=True,
                        subifd=True,
                        xres=1000 / 0.245,
                        yres=1000 / 0.245,
                        page_height=image_height)
        del final_img
        slide.close()
