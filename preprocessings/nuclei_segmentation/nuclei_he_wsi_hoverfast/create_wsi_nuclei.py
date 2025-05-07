import pyvips

import argparse
import pandas as pd
import torch
import numpy as np

import os
from tqdm import tqdm
import shutil

from pathlib import Path
from shapely import Polygon, STRtree, MultiPolygon
from shapely.geometry import box

from slidevips import SlideVips
from slidevips.ome_metadata import adapt_ome_metadata
from slidevips.torch_datasets import SlideDataset
from torch.utils.data import Dataset
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
        self.tile_size = tile_size
        self.disk_shape = 4

    def __len__(self):
        return len(self.tiles_shapely)

    def __getitem__(self, idx):
        tile_shapely = self.tiles_shapely[idx]
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
            image = watershed(
                -image, image, mask=binary_dilation(image>0, footprint=disk(self.disk_shape)),
                watershed_line=False
            )
            boundaries = find_boundaries(image, mode='outer').astype(image.dtype)
            image = image[::-1, :].copy()
        else:
            image = np.zeros((self.tile_size, self.tile_size), dtype=np.int32)
            boundaries = np.zeros_like(image)
        return np.dstack((image, boundaries))


def dataloader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset copy in this worker
    dataset.reset()  # Call the reset function


TILE_SIZE = 2048
CHANNEL_NAMES = ["cell", "boundary"]
BATCH_SIZE = 1
SLIDE_DATAFRAME_PATH = "/root/workdir/slide_dataframe_immucan_he.csv"
HOVERFAST_DIR = "/root/workdir/hoverfast_output"
SAVE_FOLDER = "/root/workdir/HandE_nuclei"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_row', type=int, help='Idx of the row in dataframe')
    args = parser.parse_args()
    idx_row = args.idx_row
    slide_dataframe = pd.read_csv(SLIDE_DATAFRAME_PATH)
    row = slide_dataframe.iloc[idx_row]
    slide_name = row["in_slide_name"]
    slide = SlideVips(row["in_slide_path"])
    resolution = slide.mpp
    magnification = slide.magnification
    slide_dim = slide.dimensions
    slide.close()

    out_slide_name = slide_name.replace(".ome", "") + ".ome.tiff"
    output_path = str(Path(SAVE_FOLDER) / out_slide_name)

    data_path = str(Path(HOVERFAST_DIR) / (slide_name + ".json.gz"))
    data = read_json_gz(data_path)

    polygons = []
    for data_item in data:
        coords = data_item['geometry']['coordinates'][0]
        polygon = Polygon(coords)
        polygons.append(polygon)
    
    stree = STRtree(polygons)
    centroids = np.asarray(
        [[polygon.centroid.x, polygon.centroid.y] for polygon in polygons])

    tile_positions = get_tiles(slide_dim[0], slide_dim[1], TILE_SIZE)
    idxs_keep = []
    for idx, tile_position in enumerate(tile_positions):
        x, y = tile_position
        tile_shapely = box(x, y, x + TILE_SIZE, y + TILE_SIZE)
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

    nuclei_dataset = NucleiDataset(polygons, tile_positions, TILE_SIZE)
    num_workers = 0
    nuclei_dataloader = torch.utils.data.DataLoader(
            nuclei_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=False,
            worker_init_fn=dataloader_worker_init_fn, pin_memory=False)


    tile_size_x = TILE_SIZE
    tile_size_y = TILE_SIZE
    n_channels = len(CHANNEL_NAMES)


    image_pyvips = pyvips.Image.black(slide_dim[0], slide_dim[1], bands=n_channels).cast("int")

    for idx_batch, out_batch in tqdm(enumerate(tqdm(nuclei_dataloader))):
        out_batch = out_batch.numpy()
        
        tile_positions_batch = tile_positions[idx_batch * BATCH_SIZE: (idx_batch + 1) * BATCH_SIZE]
        for tile_position, out in zip(tile_positions_batch, out_batch):
            tile = pyvips.Image.new_from_array(out)
            image_pyvips = image_pyvips.insert(tile,
                tile_position[0], 
                tile_position[1])
            

    del tile, out_batch
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
