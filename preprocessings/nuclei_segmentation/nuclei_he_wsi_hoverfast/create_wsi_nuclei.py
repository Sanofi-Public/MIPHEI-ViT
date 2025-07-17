import pyvips

import pandas as pd
import torch
import numpy as np

from tqdm import tqdm

from pathlib import Path
from shapely import Polygon, STRtree
from shapely.geometry import box

from slidevips import SlideVips
from slidevips.ome_metadata import adapt_ome_metadata
from torch.utils.data import Dataset
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries

import gzip
import json
import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
import rasterio
import gc
import ome_types


def read_json_gz(file_path):
    # Open the .gz file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read the decompressed content
        json_content = f.read()
        # Parse the JSON content
        data = json.loads(json_content)
        return data


def get_tiles(image_width, image_height, tile_size):
    # Generate coordinates for the starting points of the tiles
    x_coords = np.arange(0, image_width, tile_size)
    y_coords = np.arange(0, image_height, tile_size)

    # Create a mesh grid of coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten the coordinate arrays
    xx = xx.flatten()
    yy = yy.flatten()

    tile_positions = np.stack((xx, yy), axis=-1)

    return tile_positions


def order_tile_pos_per_row(tile_positions):
    xs_unique = np.unique(tile_positions[..., 0])
    tile_pos_rows_dict = {}
    for x in xs_unique:
        tile_positions_x = tile_positions[tile_positions[..., 0] == x]
        idxs_argsort = np.argsort(tile_positions_x[..., 1])
        tile_positions_x = tile_positions_x[idxs_argsort]
        tile_pos_rows_dict[x] = tile_positions_x
    return tile_pos_rows_dict


class NucleiDataset(Dataset):
    def __init__(self, nuclei_polygons, tile_positions, tile_size):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            input_size (int): Size of each input tensor.
            num_classes (int): Number of classes for the labels.
        """
        self.nuclei_polygons = nuclei_polygons
        self.stree = STRtree(self.nuclei_polygons)
        self.tiles_shapely = [box(x, y, x + tile_size, y + tile_size) for x, y in tile_positions]
        self.tile_positions = tile_positions
        self.tile_size = tile_size
        self.disk_shape = 4

    def __len__(self):
        return len(self.tiles_shapely)

    def __getitem__(self, idx):
        tile_shapely = self.tiles_shapely[idx]
        tile_position = self.tile_positions[idx]
        idxs = self.stree.query(tile_shapely)
        if len(idxs) > 0:
            polygons_roi = [self.nuclei_polygons[idx] for idx in idxs]
            shapes = [(geom, label + 1) for geom, label in zip(polygons_roi, idxs)]
            minx, miny, maxx, maxy = tile_shapely.bounds
                
            # Rasterize the shapes
            image = rasterize(
                shapes,
                out_shape=(self.tile_size, self.tile_size),
                transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, self.tile_size, self.tile_size),
                fill=0,
                dtype=np.int32
            )
            binary = image > 0
            dilated_mask = binary_dilation(binary, footprint=disk(self.disk_shape))
            distance = ndi.distance_transform_edt(~binary)
            image = watershed(-distance, markers=image, mask=dilated_mask, watershed_line=False)
            boundaries = find_boundaries(image, mode='outer').astype(image.dtype)
            image = image[::-1, :].copy()
        else:
            image = np.zeros((self.tile_size, self.tile_size), dtype=np.int32)
            boundaries = np.zeros_like(image)
        return {"mask": np.dstack((image, boundaries)), "tile_position": tile_position}


CHANNEL_NAMES = ["cell", "boundary"]


def create_wsi_nuclei(slide_path, hoverfast_dir, save_folder, tile_size=2048, batch_size=1):


    slide_name = Path(slide_path).stem
    slide = SlideVips(slide_path)
    resolution = slide.mpp
    magnification = slide.magnification
    slide_dim = slide.dimensions
    slide.close()

    out_slide_name = slide_name.replace(".ome", "") + ".ome.tiff"
    output_path = str(Path(save_folder) / out_slide_name)

    data_path = str(Path(hoverfast_dir) / (slide_name + ".json.gz"))
    data = read_json_gz(data_path)

    polygons = []
    for data_item in data:
        coords = data_item['geometry']['coordinates'][0]
        polygon = Polygon(coords)
        polygons.append(polygon)
    
    stree = STRtree(polygons)

    tile_positions = get_tiles(slide_dim[0], slide_dim[1], tile_size)
    idxs_keep = []
    for idx, tile_position in enumerate(tile_positions):
        x, y = tile_position
        tile_shapely = box(x, y, x + tile_size, y + tile_size)
        idxs = stree.query(tile_shapely)
        if len(idxs) > 0:
            idxs_keep.append(idx)
    #tile_positions = tile_positions[idxs_keep]

    tile_pos_rows_dict = order_tile_pos_per_row(tile_positions)
    xs_ordered = sorted(list(tile_pos_rows_dict.keys()))
    tile_positions = []
    for x in xs_ordered:
        tile_positions.append(tile_pos_rows_dict[x])
    tile_positions = np.vstack(tile_positions)

    nuclei_dataset = NucleiDataset(polygons, tile_positions, tile_size)
    num_workers = 0
    nuclei_dataloader = torch.utils.data.DataLoader(
        nuclei_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        pin_memory=False)


    n_channels = len(CHANNEL_NAMES)


    image_pyvips = pyvips.Image.black(slide_dim[0], slide_dim[1], bands=n_channels).cast("int")

    for idx_batch, batch in tqdm(enumerate(tqdm(nuclei_dataloader))):
        mask_batch = batch["mask"].numpy()
        tile_positions_batch = batch["tile_position"]
        
        for tile_position, mask in zip(tile_positions_batch, mask_batch):
            tile = pyvips.Image.new_from_array(mask)
            image_pyvips = image_pyvips.insert(tile,
                tile_position[0], 
                tile_position[1])
            

    del tile, mask_batch
    del nuclei_dataloader, nuclei_dataset
    gc.collect()


    image_pyvips = pyvips.Image.arrayjoin(image_pyvips.bandsplit(), across=1)
    image_pyvips = image_pyvips.cast("int").colourspace("b-w")
    ome_xml_metadata = adapt_ome_metadata(image_pyvips, resolution, CHANNEL_NAMES, magnification)
    image_height = image_pyvips.height // 2  # two channels
    xml_config = ome_types.from_xml(ome_xml_metadata)

    xml_config.images[0].pixels.type = "int32"
    ome_xml_metadata = xml_config.to_xml()

    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    image_pyvips.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    image_pyvips.tiffsave(
        output_path,
        compression="deflate",
        predictor="none",
        region_shrink='nearest',
        pyramid=True,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        subifd=True,
        xres=1000 / resolution,
        yres=1000 / resolution,
        page_height=image_height)
    del image_pyvips


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_path", type=str, required=True, help="Path to the input slide.")
    parser.add_argument("--hoverfast_dir", type=str, required=True, help="Directory containing HoverFast JSON files.")
    parser.add_argument("--save_folder", type=str, required=True, help="Directory to save the output OME-TIFF file.")
    parser.add_argument("--tile_size", type=int, default=2048, help="Size of the tiles to process.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing tiles.")
    args = parser.parse_args()

    create_wsi_nuclei(
        slide_path=args.slide_path,
        hoverfast_dir=args.hoverfast_dir,
        save_folder=args.save_folder,
        tile_size=args.tile_size,
        batch_size=args.batch_size
    )
    print(f"Slide: {args.slide_path}, HoverFast dir: {args.hoverfast_dir}, Save folder: {args.save_folder}")