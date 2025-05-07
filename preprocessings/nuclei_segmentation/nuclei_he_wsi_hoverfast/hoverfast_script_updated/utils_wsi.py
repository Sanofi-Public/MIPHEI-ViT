import numpy as np
import multiprocessing
from shapely.geometry import Polygon
import gzip
import ujson
from functools import partial
from PIL import Image
import torch
from .hoverfast import HoverFast
import scipy.ndimage as ndi
import cv2
import math
from skimage.segmentation import watershed
from shapely.validation import make_valid
from tqdm import tqdm
import os
from skimage.measure import regionprops
import logging
import datetime
import glob
import time
from skimage.color import rgb2hed, hed2rgb
from .utils_stain_deconv import *


from slidevips import SlideVips

MODE = "IF"

CHANNEL_IDXS = [0]


def magnification_from_mpp(mpp): 
    """
    Find the magnification from the micron per pixel value.

    Parameters:
    mpp (float): Micron per pixel value.

    Returns:
    float: Calculated magnification.
    """
    return 40*2**(np.round(np.log2(0.2425/mpp)))

def load_model(model_path, device):
    """
    Load the pre-trained model from the given path.

    Parameters:
    model_path (str): Path to the pre-trained model.
    device (torch.device): Device to load the model on.

    Returns:
    torch.nn.Module: Loaded model ready for inference.
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = HoverFast(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                      padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
                      up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"], conv_block=checkpoint["conv_block"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_dict"])
    model = model.half()  # Convert the model to float16 (half precision) for faster inference
    model.eval()
    return model

def divide_batch(l, n):
    """
    Split data into batches.

    This helper function divides a list into smaller batches of a specified size.

    Parameters:
    l (list): The list to be divided into batches.
    n (int): The size of each batch.

    Yields:
    list: A batch of the input list with size `n`. The last batch may be smaller if there are fewer than `n` elements remaining.
    """ 

    for i in range(0, len(l), n):  
        yield l[i:i + n]

def rgba2rgb(img):
    """
    Convert an RGBA image to an RGB image by merging the alpha channel with a white background.

    Parameters:
    img (PIL.Image.Image): An RGBA image.

    Returns:
    PIL.Image.Image: An RGB image with the alpha channel merged with a white background.
    """
    bg_color = "#" + "ffffff"
    thumb = Image.new("RGB", img.size, bg_color)
    thumb.paste(img, None, img)
    return thumb

def init_pool_processes(the_lock):
    """
    Initialize the lock for multiprocessing pool processes.

    This function sets a global lock variable for use in multiprocessing pool processes.

    Parameters:
    the_lock (multiprocessing.Lock): A multiprocessing lock to be used globally within pool processes.
    """
    global mlock
    mlock = the_lock

def multiproc(function,arg_list,n_process,output=True):
    """
    Execute a function in parallel using multiprocessing.

    This function creates a multiprocessing pool with the specified number of processes and applies the function to the argument list in parallel.

    Parameters:
    function (callable): The function to be executed in parallel.
    arg_list (list): A list of arguments to be passed to the function.
    n_process (int): The number of processes to use for multiprocessing.
    output (bool, optional): If True, the output of the function calls is returned. Default is True.

    Returns:
    list: A list of results from the function calls if output is True.
          The results are concatenated if they are lists.
    """
    pool = multiprocessing.Pool(n_process)
    out = list(pool.imap(function,arg_list))
    pool.close()
    pool.join()
    if not output:
        return
    return sum(out,[]) if isinstance(out[0],list) else out

def find_regions(mask_dir, slide_data):
    """
    Find regions in a whole slide image (WSI) for inference.

    This function either creates a tissue mask from the WSI or loads a binary mask from the specified directory.
    It then identifies regions in the slide to be processed.

    Parameters:
    mask_dir (str or None): Directory containing quality control masks. If None, a tissue mask is created from the WSI.
    slide_data (dict): Dictionary containing slide metadata and parameters.

    Returns:
    numpy.ndarray: Array of coordinates for regions to be inferred on.
    """

    if mask_dir is None:
        # create a tissue mask with simple theshold on wsi
        osh = SlideVips(os.path.join(slide_data['fpath'],slide_data['sname']+f".{slide_data['format']}"),
                        mode=MODE, channel_idxs=CHANNEL_IDXS)
        level_downsamples = osh.level_downsamples
        level = np.argwhere(np.array(level_downsamples)-32 <= 10**-2).reshape(-1)[-1]
        upscale_factor = level_downsamples[level]
        mask = Image.fromarray(
            osh.read_region((slide_data['xb'], slide_data['yb']), level,
                            (round(slide_data['width']/upscale_factor),round(slide_data['height']/upscale_factor)))).convert('L')
        mask = cv2.adaptiveThreshold(np.asarray(mask),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        osh.close()
    else:
        # load binary mask
        mask = Image.open(os.path.join(mask_dir,slide_data['sname']+'.png')).convert('L')
        upscale_factor = round(slide_data['width'] / mask.size[0])
    # gather coordinate of tissues
    pts = np.argwhere(mask)
    pts=pts*upscale_factor
    # find regions to infer on
    xbins=math.ceil((slide_data['width']-slide_data['stride_at_base'])/slide_data['tile_at_base'])
    ybins=math.ceil((slide_data['height']-slide_data['stride_at_base'])/slide_data['tile_at_base'])
    density,xbins,ybins=np.histogram2d(pts[:,1], pts[:,0], bins=[xbins,ybins],
                    range=[[int(slide_data['stride_at_base']//2), xbins*slide_data['tile_at_base']+int(slide_data['stride_at_base']//2)], [int(slide_data['stride_at_base']//2), ybins*slide_data['tile_at_base']+int(slide_data['stride_at_base']//2)]])
    return np.argwhere(density)*slide_data['tile_at_base']

def predict_ihc(regions, model, device):
    """
    Perform nuclei detection with stain deconvolution on regions using a pre-trained model.

    This function transfers regions to the GPU, performs nuclei detection using the model,
    and processes the output to return binary masks and feature maps.

    Parameters:
    regions (numpy.ndarray): Array of regions to be processed.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    tuple:
        output_cpu (numpy.ndarray): Binary masks indicating detected nuclei.
        maps_final (numpy.ndarray): Feature maps for further processing.
    """
    # Transfer regions to GPU as a torch tensor
    if len(regions.shape) == 3:
        regions = np.expand_dims(regions, axis=-1)
    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)

    regions_gpu = regions_gpu / 255

    # Convert to hed
    hed_batch = rgb_to_hed_torch(regions_gpu, device)

    # Extract Hematoxylin channel
    regions_hematoxylin = extract_h_channel_and_stack(hed_batch)

    # Convert HED back to RGB
    reconstructed_rgb_batch = hed_to_rgb_torch(regions_hematoxylin, device)

    # Scale and transpose
    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)

    # Model prediction
    output, maps = model(regions_gpu)
    
    # Post-processing
    output_processed = output.argmax(axis=1).type(torch.bool)
    
    # Move the final results to CPU and convert to numpy
    output_cpu = output_processed.detach().cpu().numpy()
    maps_cpu = maps.detach().cpu().numpy()
    maps_final = maps_cpu.astype(np.float32)

    return output_cpu,maps_final


def predict(regions, model, device):
    """
    Perform nuclei detection on regions using a pre-trained model.

    This function transfers regions to the GPU, performs nuclei detection using the model,
    and processes the output to return binary masks and feature maps.

    Parameters:
    regions (numpy.ndarray): Array of regions to be processed.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    tuple:
        output_cpu (numpy.ndarray): Binary masks indicating detected nuclei.
        maps_final (numpy.ndarray): Feature maps for further processing.
    """
    # Transfer regions to GPU as a torch tensor
    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)

    # Scale and transpose
    regions_gpu = regions_gpu.permute(0, 3, 1, 2) / 255

    # Model prediction
    output, maps = model(regions_gpu)
    
    # Post-processing
    output_processed = output.argmax(axis=1).type(torch.bool)
    
    # Move the final results to CPU and convert to numpy
    output_cpu = output_processed.detach().cpu().numpy()
    maps_cpu = maps.detach().cpu().numpy()
    maps_final = maps_cpu.astype(np.float32)

    return output_cpu,maps_final

def post_processing(output,slide_data,features_queue):
    """
    Post-process the model output to extract nuclei features and save them.

    This function performs watershed segmentation and extracts features from the model output.
    The features are then saved to the provided queue.

    Parameters:
    output (list): List of tuples containing model output masks, feature maps, and region coordinates.
    slide_data (dict): Dictionary containing slide metadata and parameters.
    features_queue (multiprocessing.Queue): Queue to store the extracted features.

    Returns:
    int: Total number of nuclei detected.
    """
    out=0
    for output_mask,maps,region_coord in output:
        dist, marker, opening = pre_watershed(output_mask,maps)
        if marker is None:
            continue
        out += region_feature(output_mask,region_coord,dist, marker, opening,slide_data,features_queue)
    return out

def pre_watershed(output_mask,maps):
    """
    Prepare for watershed segmentation by processing model output maps.

    This function normalizes the horizontal and vertical gradient maps, applies Sobel operations,
    and prepares the distance transform and marker image for watershed segmentation.

    Parameters:
    output_mask (numpy.ndarray): Binary mask indicating detected nuclei.
    maps (numpy.ndarray): Feature maps from the model output.

    Returns:
    tuple:
        dist (numpy.ndarray): Distance transform for watershed segmentation.
        marker (numpy.ndarray): Marker image for watershed segmentation.
        opening (numpy.ndarray): Processed mask for watershed segmentation.
    """

    if np.all(output_mask == 0):
        return None, None, None

    # Normalizing maps
    h_dir = cv2.normalize(maps[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(maps[1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Sobel operations
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=5)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=5)
    del h_dir, v_dir

    # Normalizing sobel results and calculating overall gradient magnitude
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    overall = np.sqrt(sobelh ** 2 + sobelv ** 2)
    del sobelh, sobelv

    # Using the original kernel size
    opening = output_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=2)


    np.subtract(overall, 1 - opening, out=overall)  # In-place operation
    np.maximum(overall, 0, out=overall)  # In-place operation

    # Preparing for watershed
    dist = (1.0 - overall) * opening
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    overall = (overall >= 0.4).astype(np.int32)

    marker = opening - overall
    np.maximum(marker, 0, out=marker)  # In-place operation
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    marker, _ = ndi.label(marker)
    del overall

    return dist, marker, opening

def watershed_object(rg,dist,submarker,opening,offset,region_coord,slide_data):
    """
    Perform watershed segmentation on detected objects.

    This function applies watershed segmentation to divide detected objects and extract their features.
    The features are then processed and saved.

    Parameters:
    rg (skimage.measure._regionprops.RegionProperties): Region properties of the detected object.
    dist (numpy.ndarray): Distance transform for watershed segmentation.
    submarker (numpy.ndarray): Marker image for watershed segmentation.
    opening (numpy.ndarray): Processed mask for watershed segmentation.
    offset (tuple): Offset coordinates for the region.
    region_coord (numpy.ndarray): Coordinates of the region.
    slide_data (dict): Dictionary containing slide metadata and parameters.

    Returns:
    list: A list of serialized polygons representing detected nuclei.
    """
    output = []
    vals = np.unique(submarker)
    vals = vals[np.nonzero(vals)]
    if vals.size > 1: # more than ones cells in the connex mask
        label = watershed(dist, markers=submarker, mask=opening) # divide the cells
    else:
        label = rg.image.astype(np.uint8)
        vals = [1]
    
    for val in vals:
        # get contour coordinates
        cell = np.uint8(label==val)
        c = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)[0][0]
        if slide_data['poly_simplification'] != 0:
            c = cv2.approxPolyDP(c,slide_data['poly_simplification']*cv2.arcLength(c,True)/1000,True)
        
        # apply threshold
        if cv2.contourArea(c) <= slide_data['threshold']/slide_data['downfactor']**2:
            continue

        # make the contour shapely compatible
        poly = Polygon(c.squeeze())
        if not poly.is_valid:
            poly = make_valid(poly)
            while poly.geom_type != 'Polygon':
                poly = poly.geoms[np.argmax([p.area for p in poly.geoms])]
            bound = poly.boundary
            if bound.geom_type != 'LineString':
                bound = bound.geoms[np.argmax([p.length for p in bound.geoms])]
            c = np.array(bound.coords[:],int)

        #get centroid coordinates
        M = cv2.moments(c)
        #ensure that area different from 0
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # remove cells in the border of the regions (outside the tile)
        if np.any(np.abs(np.array([cx,cy])-slide_data['region_size']//2)>slide_data['region_size']//2-slide_data['stride']//2):
            continue
        output.append(save_poly((c*slide_data['downfactor'])+region_coord,slide_data['downfactor']*np.array([cx,cy])+region_coord))
    return output

def region_feature(output_mask,region_coord,dist, marker, opening, slide_data, features_queue):
    """
    Extract features from each detected region and save them.

    This function isolates detected objects, performs watershed segmentation,
    and extracts features for each object. The features are then saved to the provided queue.

    Parameters:
    output_mask (numpy.ndarray): Binary mask indicating detected nuclei.
    region_coord (numpy.ndarray): Coordinates of the region.
    dist (numpy.ndarray): Distance transform for watershed segmentation.
    marker (numpy.ndarray): Marker image for watershed segmentation.
    opening (numpy.ndarray): Processed mask for watershed segmentation.
    slide_data (dict): Dictionary containing slide metadata and parameters.
    features_queue (multiprocessing.Queue): Queue to store the extracted features.

    Returns:
    int: Total number of features extracted.
    """
    output = []
    
    # isolate each detected objects
    rgs = regionprops(ndi.label(output_mask)[0])
    
    for rg in rgs:
        if rg.area < slide_data['threshold']/slide_data['downfactor']**2:
            continue
        ymin,xmin,ymax,xmax = rg.bbox

        output += watershed_object(rg,dist[ymin:ymax,xmin:xmax],marker[ymin:ymax,xmin:xmax]*rg.image,opening[ymin:ymax,xmin:xmax],(xmin,ymin),region_coord,slide_data)
    if len(output)!=0:
        features_queue.put(",\n".join(output))
    return len(output)

def save_poly(poly,centroid,object_class = {'name': 'Nuclei', 'colorRGB': -65536}):
    """
    Serialize a polygon representing a detected object.

    This function converts a polygon and its associated features into a serialized format
    compatible with GeoJSON.

    Parameters:
    poly (numpy.ndarray): Coordinates of the polygon.
    centroid (numpy.ndarray): Centroid of the polygon.
    object_class (dict, optional): Classification information for the object. Default is {'name': 'Nuclei', 'colorRGB': -65536}.

    Returns:
    str: Serialized polygon in GeoJSON format.
    """
    feature = {}
    feature["geometry"] = {'type':'Polygon','coordinates':(tuple(map(tuple,poly.squeeze()))+(tuple(poly[0].squeeze()),),)}
    feature["geometry"]["centroid"] = [ int(coord) for coord in centroid]
    feature["properties"] = {'object_type': 'cell',
                                'classification': object_class,
                                'isLocked': False}
    feature["type"] = "Feature"
    return ujson.dumps(feature)

def writer(features_queue,output_path):
    """
    Save detected features to a JSON file.

    This function reads features from the provided queue and saves them to a compressed JSON file.

    Parameters:
    features_queue (multiprocessing.Queue): Queue containing the extracted features.
    output_path (str): Path to the output JSON file.
    """
    # receive all detection from the worker and save them in a json file
    with gzip.open(output_path, 'wt', encoding="utf-8") as file:
            file.write('[')
            first = True
            while True:
                try:
                    feature = features_queue.get()
                except:
                    continue
                if feature is None: # end of analysis
                    break
                file.write(","*(not first)+'\n'+feature)
                first = False
            file.write('\n]')

def preload_batch(batch_coords,slide_data,n_process):
    """
    Preload a batch of regions into RAM.

    This function preloads a batch of regions into RAM using multiprocessing.

    Parameters:
    batch_coords (numpy.ndarray): Coordinates of the regions to be preloaded.
    slide_data (dict): Dictionary containing slide metadata and parameters.
    n_process (int): Number of processes to use for multiprocessing.

    Returns:
    numpy.ndarray: Array of preloaded regions.
    """
    # preload a batch into the ram
    osh = SlideVips(os.path.join(slide_data['fpath'],slide_data['sname']+f".{slide_data['format']}"), mode=MODE, channel_idxs=CHANNEL_IDXS)
    positions = batch_coords.copy()
    positions[..., 0] += slide_data['xb']
    positions[..., 1] += slide_data['yb']
    level = slide_data["level"]
    tile_size = (int(slide_data['region_size']*(slide_data['working_d']/slide_data['downfactor'])),)*2
    regions = osh.read_regions(positions, level, tile_size)
    osh.close()
    resize_size = (slide_data['region_size'],)*2
    if slide_data['working_d'] != slide_data['downfactor']:
        regions = np.asarray([cv2.resize(region, resize_size, interpolation=cv2.INTER_LINEAR) for region in regions])
    return regions

def processing(batch_coords,slide_data,model,device,batch_to_gpu,n_process, stain):
    """
    Process a batch of regions for nuclei detection.

    This function processes a batch of regions for nuclei detection using a pre-trained model.

    Parameters:
    batch_coords (numpy.ndarray): Coordinates of the regions to be processed.
    slide_data (dict): Dictionary containing slide metadata and parameters.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).
    batch_to_gpu (int): Target batch size for GPU.
    n_process (int): Number of processes to use for multiprocessing.

    Returns:
    list: List of tuples containing output masks, feature maps, and region coordinates.
    """
    loaded_batch = preload_batch(batch_coords,slide_data,n_process)
    arg_list1 = []
    for regions in tqdm(divide_batch(loaded_batch,batch_to_gpu),desc="inner",leave=False, total = math.ceil(len(loaded_batch)/batch_to_gpu)):
        
        if stain == "ihc_dab":
            output_mask, maps = predict_ihc(regions, model, device)
        else:
            output_mask, maps = predict(regions, model, device)
        
        arg_list1 += [(output_mask[j], maps[j]) for j in range(len(output_mask))]
    torch.cuda.empty_cache()
    return list(map(tuple.__add__, arg_list1, map(lambda x:(x,) ,list(batch_coords))))

def infer_on_batches(batch_coords,slide_data,model,device,batch_to_gpu,n_process,stain, features_queue):
    """
    Perform nuclei detection on batches of regions.

    This function processes batches of regions for nuclei detection using a pre-trained model,
    and extracts features from the detected nuclei.

    Parameters:
    batch_coords (numpy.ndarray): Coordinates of the regions to be processed.
    slide_data (dict): Dictionary containing slide metadata and parameters.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).
    batch_to_gpu (int): Target batch size for GPU.
    n_process (int): Number of processes to use for multiprocessing.
    features_queue (multiprocessing.Queue): Queue to store the extracted features.

    Returns:
    int: Total number of features extracted.
    """
    arg_list1 = processing(batch_coords,slide_data,model,device,batch_to_gpu,n_process, stain)
    pool = multiprocessing.Pool(processes=n_process)
    results = list(pool.imap(partial(post_processing,slide_data=slide_data, features_queue=features_queue),divide_batch(arg_list1,int(np.ceil(len(arg_list1)/n_process)))))
    pool.close()
    pool.join()
    return sum(results)

def get_slide(sname,sformat,fpath,mag,kernel_size,region_size,threshold,outdir,poly_simplify_tolerance,logger):
    """
    Gather data for a specific slide and set up parameters for processing.

    This function retrieves metadata and initializes parameters for a given slide,
    preparing it for nuclei detection.

    Parameters:
    sname (str): Slide name.
    sformat (str): Slide format (e.g., 'tif').
    fpath (str): File path to the slide.
    mag (float): Target magnification.
    kernel_size (int): Size of the kernel for processing.
    region_size (int): Size of the region to be processed.
    threshold (float): Minimum size threshold for nuclei area in square micrometers.
    outdir (str): Output directory.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    logger (logging.Logger): Logger for logging messages.

    Returns:
    dict: A dictionary containing slide data and processing parameters."""

    slide_data = {}
    
    #Slide information
    slide_data['format'] = sformat
    slide_data['sname'] = sname
    slide_data['fpath'] = fpath

    #Slide data
    osh = SlideVips(os.path.join(slide_data['fpath'],slide_data['sname']+f".{slide_data['format']}"),
                    mode=MODE, channel_idxs=CHANNEL_IDXS)
    slide_data['xb'] = 0
    slide_data['yb'] = 0
    slide_data['width'] = osh.level_dimensions[0][0]
    slide_data['height'] = osh.level_dimensions[0][1]
    mpp_value = osh.mpp
    
    #Checks for mpp value
    if mpp_value is None:
        mpp = 0.245  # Default value
        warning_message = "WARNING. The MPP value was not found in openslide properties; using default value of 0.245."
        print(warning_message)
        logger.warning(warning_message)
    else:
        mpp = float(mpp_value)

    #Base maggnification
    slide_data['base_mag'] = base_mag = magnification_from_mpp(mpp)

    if base_mag < mag:
        error_message = f"ERROR: Base magnification level lower than {mag}X detected. Only {mag}X slides (~{mpp}mpp) are compatible with HoverFast."
        logger.error(error_message)
        raise ValueError(error_message)
    slide_data['mpp'] = mpp

    #Downfactor of working magnification
    slide_data['downfactor'] = downfactor = base_mag/mag

    #Find correct level
    
    level_downsamples = np.array(osh.level_downsamples,int)
    osh.close()
    slide_data['level'] = level = np.argwhere(level_downsamples <= downfactor).reshape(-1)[-1]
    slide_data['working_d'] = level_downsamples[level]
    
    #Model kernel size
    slide_data['kernel_size'] = kernel_size

    #Stride size for overlapping region
    slide_data['stride'] = slide_data['kernel_size']//2
    slide_data['stride_at_base'] = int(slide_data['stride']*downfactor)
    
    #Region size
    slide_data['region_size'] = region_size
    slide_data['region_at_base'] = int(slide_data['region_size']*downfactor)

    #Tile is the part of the region that are being processed
    slide_data['tile_size'] = region_size - slide_data['stride']
    slide_data['tile_at_base'] = int(slide_data['tile_size']*downfactor)

    #Output parameters
    slide_data['poly_simplification'] = poly_simplify_tolerance
    slide_data['threshold'] = threshold/(mpp**2)
    slide_data['outdir'] = outdir

    return slide_data

def infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_on_mem,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain,logger):
    """
    Perform nuclei detection on a whole slide image (WSI).

    This function processes a WSI for nuclei detection by dividing the slide into regions,
    performing inference using a pre-trained model, and saving the detected features.

    Parameters:
    sname (str): Slide name.
    sformat (str): Slide format (e.g., 'tif').
    fpath (str): File path to the slide.
    mask_dir (str or None): Directory containing quality control masks. If None, a tissue mask is created from the WSI.
    outdir (str): Output directory.
    mag (float): Target magnification.
    batch_on_mem (int or None): Number of regions to load into memory at a time. Default is the total number of regions.
    batch_to_gpu (int): Target batch size for GPU.
    region_size (int): Size of the region to be processed.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).
    n_process (int): Number of processes to use for multiprocessing.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    threshold (float): Minimum size threshold for nuclei area in square micrometers.
    logger (logging.Logger): Logger for logging messages.

    Returns:
    tuple: Total number of regions processed and total number of nuclei detected.

    """

    #UNet model data
    kernel_size = 256

    #gather data of the slide
    slide_data = get_slide(sname,sformat,fpath,mag,kernel_size,region_size,threshold,outdir,poly_simplify_tolerance,logger)
    
    #find regions to infer on
    coords = find_regions(mask_dir,slide_data)
    
    print(f"|- Computation done on {len(coords)} patches.")
    #infer on regions by batches
    size=0

    if batch_on_mem is None:
        batch_on_mem = coords.shape[0]
    
    size = 0
    features_queue = multiprocessing.Manager().Queue()
    writer_process = multiprocessing.Process(target=writer, args=(features_queue, os.path.join(outdir, sname + ".json.gz")))
    writer_process.start()
    for batch_coords in tqdm(divide_batch(coords,batch_on_mem),desc="outer",leave=False,total=math.ceil(len(coords)/batch_on_mem)):
        size += infer_on_batches(batch_coords,slide_data,model,device,batch_to_gpu,n_process,stain,features_queue)
    features_queue.put(None)
    writer_process.join()
    return len(coords),size


def main_wsi(args) -> None:
    """
    Main entry point for nuclei detection on whole slide images (WSI).

    This function parses the command-line arguments, sets up the logger, loads the pre-trained model,
    and processes each slide for nuclei detection.

    Parameters:
    args (argparse.Namespace): Command-line arguments.
    """
    
    #get args

    slide_dirs = args.slide_folder
    outdir = args.outdir
    mask_dir = args.binmask_dir
    mag = args.magnification
    batch_to_gpu = args.batch_gpu
    batch_on_mem = (args.batch_mem//batch_to_gpu)*batch_to_gpu
    region_size = args.tile_size
    n_process = args.n_process
    model_path = args.model_path
    poly_simplify_tolerance = args.poly_simplify
    threshold = args.size_threshold
    stain = args.stain

    if n_process is None:
        n_process = os.cpu_count()

    os.mkdir(outdir) if not os.path.exists(outdir) else None

    #config logger

    logger = logging.getLogger(f"{outdir}/HoverFast_log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"))

    f_handler = logging.FileHandler(f"{outdir}/HoverFast_log_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")+".log")
    c_handler = logging.StreamHandler()

    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=True
    model = load_model(model_path,device)

    #get input files
    if len(slide_dirs)==1:
        #input is a glob pattern
        slide_dirs = glob.glob(slide_dirs[0])

    if not slide_dirs:
        error_message = "No slides detected in the specified directory."
        logger.error(error_message)
        raise ValueError(error_message)

    stats={}

    for slide_dir in slide_dirs:
        temp = os.path.basename(slide_dir).rpartition('.')
        sname, sformat = temp[0],temp[-1]
        fpath = os.path.dirname(slide_dir)
        print(f"- Working on {sname}")
        stats[sname]=[]
        #convert and create a pyramidal tiff
        
        try:
            start=time.time()
            n_patches,n_objects = infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_on_mem,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain, logger)
            stats[sname].append(n_patches)
            stats[sname].append(n_objects)
            stats[sname].append(time.time()-start)
            print(f"running time: {stats[sname][-1]}, #patches {stats[sname][0]} and #objects {stats[sname][1]}")
        except Exception as e:
            print(f"File {sname} failed with error: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"File {sname} failed: {e}", exc_info=True)
