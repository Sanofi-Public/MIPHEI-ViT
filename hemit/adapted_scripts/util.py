"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0 )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def multi_channel_to_rgb(image, channel_colors=None):
    """
    Convert a multi-channel image into an RGB image by assigning colors to each channel.

    :param image: A NumPy array of shape (H, W, C) representing the multi-channel image.
    :param channel_colors: A list of tuples (R, G, B) specifying the color for each channel.
                           Values should be in the range [0, 255].
    :return: An RGB image as a NumPy array of shape (H, W, 3).
    """
    num_channels = image.shape[-1]

    # Default colors (assign unique hues to each channel)
    if channel_colors is None:
        colormap = plt.get_cmap("tab10")
        channel_colors = [tuple(int(255 * c) for c in colormap(i)[:3]) for i in range(num_channels)]

    # Normalize image if necessary
    image = image.astype(np.float32)
    # Compute max values per channel, avoiding division by zero
    max_vals = np.max(image, axis=(0, 1), keepdims=True)
    max_vals[max_vals == 0] = 1  # Prevent division by zero for empty channels
    # Normalize image
    image = image / max_vals

    # Create an empty RGB image
    rgb_image = np.zeros((*image.shape[:2], 3), dtype=np.float32)

    # Assign colors to each channel
    for i in range(num_channels):
        color = np.array(channel_colors[i]) / 255.0  # Normalize color to [0,1]
        rgb_image += np.expand_dims(image[:, :, i], axis=-1) * color

    # Clip values to range [0,1] and convert to 8-bit
    rgb_image = np.clip(rgb_image, 0, 1) * 255
    return rgb_image.astype(np.uint8)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    if image_numpy.shape[-1] > 3:
        image_pil = Image.fromarray(multi_channel_to_rgb(image_numpy))
    else:
        image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        print('aspect_ratio > 1.0')
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        print('aspect_ratio < 1.0')
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
