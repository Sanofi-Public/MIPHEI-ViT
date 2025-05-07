from typing import Tuple, List

import cv2
import numpy as np


def get_locs_otsu(thumbnail_or_mask: np.ndarray, slide_dim: Tuple[int, int],
                           tile_size_lvl0: Tuple[int, int], tile_overlap: int = 0,
                           mask_thresh: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the positions of tiles based on Otsu thresholding.

    Args:
        thumbnail: The input thumbnail image.
        slide_dim: The dimensions of the slide.
        tile_size_lvl0: The size of the tiles at level 0.
        tile_overlap: The overlap between tiles.
        mask_thresh: The threshold for considering a tile as tissue.

    Returns:
        tile_positions: The positions of the tiles.
        tissue_percentages: The tissue percentages of the tiles.
    """
    if thumbnail_or_mask.dtype == bool:
        mask = thumbnail_or_mask
    else:
        if thumbnail_or_mask.shape[-1] > 1:
            thumbnail_1d = np.uint8(thumbnail_or_mask.std(axis=-1))
        else:
            thumbnail_1d = thumbnail_or_mask
        _, mask = cv2.threshold(thumbnail_1d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask > 0

    # Scale tile_size and tile_positions
    thumbnail_shape = np.roll(np.array(mask.shape[:2]), 1)
    scale_ratio = slide_dim / thumbnail_shape
    scaled_tile_size = tile_size_lvl0 / scale_ratio
    scaled_tile_overlap = tile_overlap / scale_ratio

    tile_positions = []
    tissue_percentages = []

    ys_thumbnail = np.arange(0, thumbnail_shape[1] + 1,
                             scaled_tile_size[1] - scaled_tile_overlap[1])
    ys = np.arange(0, slide_dim[1] + 1,
                   tile_size_lvl0 - tile_overlap)
    xs_thumbnail = np.arange(0, thumbnail_shape[0] + 1,
                             scaled_tile_size[0] - scaled_tile_overlap[0])
    xs = np.arange(0, slide_dim[0] + 1,
                   tile_size_lvl0 - tile_overlap)

    for y_thumb, y in zip(ys_thumbnail, ys):
        for x_thumb, x in zip(xs_thumbnail, xs):
            x_min, y_min = int(x_thumb), int(y_thumb)
            x_max, y_max = int(x_thumb + scaled_tile_size[0]), int(y_thumb + scaled_tile_size[1])
            tile = mask[y_min: y_max, x_min: x_max]
            if tile.size == 0:
                continue
            mask_percentage = np.count_nonzero(tile) / tile.size
            if mask_percentage > mask_thresh:
                tile_positions.append([x, y])
                tissue_percentages.append(mask_percentage)
    tile_positions = np.asarray(tile_positions)
    tissue_percentages = np.asarray(tissue_percentages)
    return tile_positions, tissue_percentages


def order_tiles_horizontally(coordinates: np.ndarray) -> List[int]:
    """
    Order the tiles by horizontal lines.

    Args:
        coordinates: The coordinates of the images.

    Returns:
        indices: The indices of the sorted coordinates.
    """
    # Sort the coordinates by y-coordinate in descending order
    sorted_coordinates = coordinates[np.argsort(coordinates[:, 1])]
    # Sort the coordinates within each horizontal line by x-coordinate in ascending order
    sorted_coordinates = sorted_coordinates[np.lexsort((sorted_coordinates[:, 0],))]
    # Get the indices of the sorted coordinates
    indices = [np.where((coordinates == c).all(axis=1))[0][0] for c in sorted_coordinates]
    return indices
