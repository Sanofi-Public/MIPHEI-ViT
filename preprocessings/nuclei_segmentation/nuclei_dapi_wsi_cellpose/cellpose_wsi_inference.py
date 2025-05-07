import pyvips
from slidevips import SlideVips
from cellpose import models
from cellpose.dynamics import compute_masks
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage as ndi
from skimage.util import map_array
from skimage.segmentation import watershed
from skimage.morphology import binary_dilation, disk
import ome_types
from slidevips.ome_metadata import adapt_ome_metadata
import os
import gc


def find_boundaries(mask):
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    boundaries = (
        (padded_mask[1:-1, 1:-1] != padded_mask[:-2, 1:-1]) |  # Up
        (padded_mask[1:-1, 1:-1] != padded_mask[2:, 1:-1]) |   # Down
        (padded_mask[1:-1, 1:-1] != padded_mask[1:-1, :-2]) |  # Left
        (padded_mask[1:-1, 1:-1] != padded_mask[1:-1, 2:])     # Right
    )
    # Diagonal shifts
    boundaries |= (
        (padded_mask[1:-1, 1:-1] != padded_mask[:-2, :-2]) |   # Up-left
        (padded_mask[1:-1, 1:-1] != padded_mask[:-2, 2:]) |    # Up-right
        (padded_mask[1:-1, 1:-1] != padded_mask[2:, :-2]) |    # Down-left
        (padded_mask[1:-1, 1:-1] != padded_mask[2:, 2:])       # Down-right
    )
    return boundaries


def normalize99(Y, x01, x99, copy=True):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image (for downsample, use [Ly x Lx] or [Lz x Ly x Lx]).
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.
        downsample (bool, optional): Whether to downsample image to compute percentiles. Defaults to False.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    X = X.astype("float32") if X.dtype!="float64" and X.dtype!="float32" else X
    X -= x01 
    X /= (x99 - x01)
    return X


def get_tile_positions(slide_dims, tile_size):
    tile_positions = np.meshgrid(
        np.arange(0, slide_dims[0], tile_size),
        np.arange(0, slide_dims[1], tile_size))
    tile_positions = np.stack(tile_positions, axis=-1).reshape((-1, 2))
    return tile_positions


class HDF5TileDataset(Dataset):
    def __init__(self, hdf5_file, tile_positions, slide_dims, tile_size=512, overlap=128):
        """
        Initialize the dataset and open the HDF5 file once.
        """
        self.hdf5_file = hdf5_file
        self.tile_positions = tile_positions
        self.slide_dims = slide_dims
        self.tile_size = tile_size
        self.overlap = overlap

        # Open the HDF5 file
        self.file = h5py.File(self.hdf5_file, 'r')
        self.dp_dset = self.file["dp"]
        self.cellprob_dset = self.file["cellprob"]
        self.counts_dset = self.file["count"]

    def __len__(self):
        return len(self.tile_positions)

    def __getitem__(self, idx):
        """
        Fetch a single item (tile and associated data) from the dataset.
        """
        y_min, x_min = self.tile_positions[idx]
        x_min_over, y_min_over = max(x_min - self.overlap, 0), max(y_min - self.overlap, 0)
        x_max, y_max = x_min + self.tile_size, y_min + self.tile_size
        x_max_over, y_max_over = min(x_max + self.overlap, self.slide_dims[0]), min(y_max + self.overlap, self.slide_dims[1])

        # Extract chunks
        dp_chunk = np.float32(self.dp_dset[:, x_min_over:x_max_over, y_min_over:y_max_over])
        cellprob_chunk = np.float32(self.cellprob_dset[x_min_over:x_max_over, y_min_over:y_max_over])
        count_chunck = np.float32(self.counts_dset[x_min_over:x_max_over, y_min_over:y_max_over])
        count_chunck[count_chunck == 0] = 1.
        dp_chunk /= count_chunck
        cellprob_chunk /= count_chunck

        cellmask_chunk = cellprob_chunk > 0.

        # Bounding boxes for tile and overlap
        bbox = np.asarray([x_min, x_max, y_min, y_max])
        bbox_over = np.asarray([x_min_over, x_max_over, y_min_over, y_max_over])

        keep_mask = np.zeros_like(cellmask_chunk, dtype=bool)
        x_min_keepmask, x_max_keepmask = x_min - x_min_over, -(x_max_over - x_max)
        y_min_keepmask, y_max_keepmask = y_min - y_min_over, -(y_max_over - y_max)
        keep_mask[x_min_keepmask: x_max_keepmask, y_min_keepmask: y_max_keepmask] = True

        return {
            "dp_chunk": dp_chunk,
            "cellmask_chunk": cellmask_chunk,
            "bbox": bbox,
            "bbox_over": bbox_over,
            "keep_mask": keep_mask
        }

    def __del__(self):
        """
        Ensure the HDF5 file is closed when the dataset is deleted.
        """
        self.file.close()


CELLPROB_THRESHOLD = 0.
MPP_TARGET = 0.325
X01 = 0.
X99 = 0.65


def cellpose_wsi_inference(slide_path, idx_dapi, cellpose_ckpt, tile_size_inference, output_dir, tile_size_overlap=128, tile_size_cleaning=16384):
    slide_name = Path(slide_path).stem
    output_slide_path = str(Path(output_dir) / f"{slide_name}").replace(".ome", "")
    output_slide_path += ".ome.tiff"
    hdf5_path = str(Path(output_dir) / f"{slide_name}.hdf5")
    Path(output_dir).mkdir(exist_ok=True)

    torch.cuda.empty_cache()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    model = models.CellposeModel(pretrained_model=cellpose_ckpt, device=torch.device("cuda"), gpu=True)


    slide_if = SlideVips(slide_path, mode="IF", channel_idxs=[idx_dapi])
    scale_factor = slide_if.mpp / MPP_TARGET
    if scale_factor != 1.:
        slide_if.resize(scale_factor)

    tile_positions = get_tile_positions(slide_if.dimensions, tile_size=tile_size_inference - tile_size_overlap)
    tile_positions_nooverlap = get_tile_positions(slide_if.dimensions, tile_size=tile_size_inference)

    slide_dims = (slide_if.dimensions[1], slide_if.dimensions[0])
    if not Path(hdf5_path).exists():
        with h5py.File(hdf5_path, 'w') as f:
            # Create a compressed dataset
            dp_dset = f.create_dataset(
                'dp',
                shape=(2, slide_dims[0], slide_dims[1]),
                dtype="float16",
                compression="lzf",
                chunks=(2, tile_size_inference, tile_size_inference),  # Chunk size
                fillvalue=0.
            )
            cellprob_dset = f.create_dataset(
                'cellprob',
                shape=slide_dims,
                dtype="float16",
                compression="lzf",
                chunks=(tile_size_inference, tile_size_inference),  # Chunk size
                fillvalue=0.
            )
            counts_dset = f.create_dataset(
                'count',
                shape=slide_dims,
                dtype="uint8",
                chunks=(tile_size_inference, tile_size_inference),  # Chunk size
                fillvalue=0
            )

            for tile_position in tqdm(tile_positions, desc="Inference CellPose", total=len(tile_positions)):
                tile = slide_if.read_region(
                    tile_position, 0, (tile_size_inference, tile_size_inference))

                x_input = np.expand_dims(tile, axis=-1) / 255

                _, flows, _ = model.eval(
                        [normalize99(x_input, X01, X99)],
                        channels=[0, 0],
                        diameter=model.diam_labels,
                        batch_size=128,
                        min_size=5,
                        normalize=False,
                        compute_masks=False
                        )
                dp = np.float16(np.squeeze(flows[0][1]))
                cellprob = np.float16(np.squeeze(flows[0][2]))
                y, x = tile_position
                x_max, y_max = min(x+tile_size_inference, slide_dims[0]), min(y+tile_size_inference, slide_dims[1])
                dp_dset[:, x:x_max, y:y_max] += dp[:, :(x_max-x), :(y_max-y)]
                cellprob_dset[x:x_max, y:y_max] += cellprob[:(x_max-x), :(y_max-y)]
                counts_dset[x:x_max, y:y_max] += np.ones_like(
                    cellprob[:(x_max-x), :(y_max-y)], dtype=np.uint8)

    gc.collect()

    #dataset = HDF5TileDataset(hdf5_path, tile_positions, slide_dims, tile_size=tile_size, overlap=128)
    dataset = HDF5TileDataset(hdf5_path, tile_positions_nooverlap, slide_dims, tile_size=tile_size_inference, overlap=tile_size_overlap)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    rescale = model.diam_mean / model.diam_labels
    interp = True
    niter = 1 / rescale * 200
    resize = None

    masks_wsi = np.zeros(slide_dims, dtype=np.uint32)
    max_cell_id = 0
    cell_ids = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Nuclei mask generation"):
        dp_chunk = data["dp_chunk"][0].numpy()
        cellmask_chunk = data["cellmask_chunk"][0].numpy()
        keep_mask = data["keep_mask"][0].numpy()
        x_min_over, x_max_over, y_min_over, y_max_over = data["bbox_over"][0].numpy()
        if np.sum(cellmask_chunk) < 10:
            continue
        masks = compute_masks(dp_chunk, cellmask_chunk, niter=niter, cellprob_threshold=0.,
                              flow_threshold=0.4, interp=interp, device=model.device)
        masks = np.uint32(masks)
        mask_tile = masks_wsi[x_min_over:x_max_over,  y_min_over:y_max_over]

        keep_cell_ids = np.unique(masks[keep_mask & (masks> 0)])
        keep_mask = np.isin(masks, keep_cell_ids) & (mask_tile == 0)
        # get cell ids in current tile and add curr max cell for consistency
        tile_cell_ids = np.unique(masks[keep_mask])
        
        masks[keep_mask] += max_cell_id
        if len(tile_cell_ids) > 0:
            tile_cell_ids += max_cell_id
            max_cell_id = tile_cell_ids.max()

        cell_ids.append(tile_cell_ids)
        mask_tile[keep_mask] = masks[keep_mask]
        masks_wsi[x_min_over:x_max_over,  y_min_over:y_max_over] = mask_tile

    gc.collect()

    input_vals, index = np.unique(masks_wsi[masks_wsi>0], return_index=True)
    input_vals = input_vals[np.argsort(index)]
    output_vals = np.arange(1, len(input_vals) +1, dtype=np.uint32)
    masks_wsi = map_array(masks_wsi, input_vals, output_vals)

    masks_boundaries_wsi = np.zeros_like(masks_wsi, dtype=bool)
    radius = 1 / MPP_TARGET
    tile_positions_mask = get_tile_positions(slide_if.dimensions, tile_size=tile_size_cleaning)
    for tile_position in tqdm(tile_positions_mask, desc="Nuclei Expansion"):
        y_min, x_min = tile_position
        x_max, y_max = min(x_min+tile_size_cleaning, slide_dims[0]), min(y_min+tile_size_cleaning, slide_dims[1])
        masks_tile = masks_wsi[x_min:x_max, y_min:y_max]
        masks_boundaries_tile = find_boundaries(masks_tile)
        masks_boundaries_wsi[x_min:x_max, y_min:y_max] = masks_boundaries_tile

        binary = masks_tile > 0
        dilated_mask = binary_dilation(binary, footprint=disk(radius))
        distance = ndi.distance_transform_edt(~binary)
        masks_tile = watershed(-distance, markers=masks_tile, mask=dilated_mask, watershed_line=False)

        masks_tile = np.uint32(masks_tile)
        masks_wsi[x_min:x_max, y_min:y_max] = masks_tile
    
    gc.collect()
    
    image_pyvips = pyvips.Image.new_from_array(masks_wsi).cast("int").colourspace("b-w")
    image_boundary_pyvips = pyvips.Image.new_from_array(np.int32(masks_boundaries_wsi)).cast("int").colourspace("b-w")
    image_pyvips = pyvips.Image.arrayjoin([image_pyvips, image_boundary_pyvips], across=1).cast("int").colourspace("b-w")
    resolution = slide_if.mpp
    ome_xml_metadata = adapt_ome_metadata(image_pyvips, resolution, ["cell", "boundary"], slide_if.magnification)
    image_height = image_pyvips.height // 2  # two channels
    xml_config = ome_types.from_xml(ome_xml_metadata)

    xml_config.images[0].pixels.type = "int32"
    ome_xml_metadata = xml_config.to_xml()

    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    image_pyvips.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml_metadata)

    image_pyvips.tiffsave(
        output_slide_path,
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
    os.remove(hdf5_path)
    return output_slide_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply cleaning to WSI")
    parser.add_argument("--slide_dataframe_path", type=str, required=True, help="Path to the input slide")
    parser.add_argument("--idx_dapi", type=int, required=True, help="Path to the input slide")
    parser.add_argument("--cellpose_ckpt_path", type=str, required=True, help="Path of the CellPose checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the nuclei slides will be saved")

    parser.add_argument("--tile_size_inference", type=int, default=8192, help="Tile size used during CellPose inference")
    parser.add_argument("--tile_size_cleaning", type=int, default=16384, help="Tile size used during nuclei mask construction and cleaning")
    args = parser.parse_args()

    slide_dataframe_path = args.slide_dataframe_path
    slide_dataframe = pd.read_csv(slide_dataframe_path)
    cellpose_ckpt_path = args.cellpose_ckpt_path
    output_dir = args.output_dir
    idx_dapi = args.idx_dapi
    tile_size_inference = args.tile_size_inference
    tile_size_cleaning = args.tile_size_cleaning

    output_paths = []
    for _, row in tqdm(slide_dataframe.iterrows(), total=len(slide_dataframe)):
        slide_path = row["targ_slide_path"]
        output_path = cellpose_wsi_inference(
            slide_path, idx_dapi, cellpose_ckpt_path, tile_size_inference, output_dir,
            tile_size_overlap=128, tile_size_cleaning=tile_size_cleaning)
        output_paths.append(output_path)
        gc.collect()

    slide_dataframe["nuclei_slide_path"] = output_paths
    slide_dataframe["nuclei_slide_path"] = slide_dataframe["nuclei_slide_path"].apply(lambda x: str(Path(x).resolve()))
    slide_dataframe.to_csv(slide_dataframe_path, index=False)
