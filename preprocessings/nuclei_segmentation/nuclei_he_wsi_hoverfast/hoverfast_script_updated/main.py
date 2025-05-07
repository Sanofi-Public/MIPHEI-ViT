import pyvips
import argparse
import torch
from .utils_wsi import *
from .training_utils import *
from .hoverfast import *
from .utils_roi import *
from . import __version__


def get_args():
    """Parsing command line arguments"""

    parser = argparse.ArgumentParser(prog="HoverFast", description='Blazing fast nuclei segmentation for H&E WSIs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version=f'HoverFast {__version__}')
    subparsers = parser.add_subparsers(title="mode", help="Three modes: infer_wsi, infer_roi, train ", dest='mode')


    ###### INFER WSI PARSER

    infer_wsi_parser = subparsers.add_parser("infer_wsi", help="Perform nuclei detection on WSI")

    
    infer_wsi_parser.add_argument('slide_folder',
                        help="input filename pattern.",
                        nargs="+",
                        type=str)
    infer_wsi_parser.add_argument('-o', '--outdir',
                        help="outputdir, default ./output/",
                        default="./output/",
                        type=str)
    infer_wsi_parser.add_argument('-b', '--binmask_dir',
                        help="quality control mask directory",
                        default=None,
                        type=str)
    infer_wsi_parser.add_argument('-m', '--model_path',
                        help="path to pre-trained model",
                        default= "./hoverfast_crosstissue_best_model.pth",
                        type=str)
    infer_wsi_parser.add_argument('-l', '--magnification',
                        help="magnification to work on",
                        default=40,
                        type=float)
    infer_wsi_parser.add_argument('-p', '--poly_simplify',
                        help="float representing the tolerance for simplifying the polygons",
                        default=6,
                        type=float)
    infer_wsi_parser.add_argument('-s', '--size_threshold',
                        help="minimum size threshold for nuclei area in square micrometer",
                        default=5,
                        type=float)
    infer_wsi_parser.add_argument('-st', '--stain',
                        help="Staining type, choose between 'he' and 'ihc_dab', default 'he'",
                        choices=['he', 'ihc_dab'],
                        default='he',
                        type=str)
    infer_wsi_parser.add_argument('-r', '--batch_mem',
                        help="maximum batches saves on memory (RAM)",
                        default=500,
                        type=int)
    infer_wsi_parser.add_argument('-g', '--batch_gpu',
                        help="Target batch size for GPU: +1 in batch ~ +2GB VRAM (for pretrain model). Avoid matching or exceeding estimated GPU VRAM.",
                        default=int(np.round(torch.cuda.mem_get_info()[1]/1024**3))//2-1 if torch.cuda.is_available() else 1,
                        type=int)
    infer_wsi_parser.add_argument('-t', '--tile_size',
                        help="region size to compute on",
                        default=1024,
                        type=int)
    infer_wsi_parser.add_argument('-n', '--n_process',
                        help="number of worker for multiprocessing, default is os.cpu_count()",
                        default=None,
                        type=int) 

    ########### INFER ROI PARSER

    infer_roi_parser = subparsers.add_parser("infer_roi", help="Perform nuclei detection on directory with ROIs")


    infer_roi_parser.add_argument('slide_folder',
                        help="input filename pattern.",
                        nargs="+",
                        type=str)
    infer_roi_parser.add_argument('-o', '--outdir',
                        help="outputdir, default ./output/",
                        default="./output/",
                        type=str)
    infer_roi_parser.add_argument('-m', '--model_path',
                        help="path to pre-trained model",
                        default= "./hoverfast_crosstissue_best_model.pth",
                        type=str)
    infer_roi_parser.add_argument('-p', '--poly_simplify',
                        help="float representing the tolerance for simplifying the polygons",
                        default=6,
                        type=float)
    infer_roi_parser.add_argument('-s', '--size_threshold',
                        help="minimum size threshold for nuclei area in square micrometer",
                        default=85,
                        type=float)
    infer_roi_parser.add_argument('-st', '--stain',
                        help="Staining type, choose between 'he' and 'ihc_dab', default 'he'",
                        choices=['he', 'ihc_dab'],
                        default='he',
                        type=str)
    infer_roi_parser.add_argument('-r', '--batch_mem',
                        help="maximum batches saves on memory",
                        default=500,
                        type=int)
    infer_roi_parser.add_argument('-g', '--batch_gpu',
                        help="Target batch size for GPU: +1 in batch ~ +2GB VRAM (for pretrain model). Avoid matching or exceeding estimated GPU VRAM.",
                        default=int(np.round(torch.cuda.mem_get_info()[1]/1024**3))//2-1 if torch.cuda.is_available() else 1,
                        type=int)
    infer_roi_parser.add_argument('-n', '--n_process',
                        help="number of worker for multiprocessing, default is os.cpu_count()",
                        default=None,
                        type=int)
    infer_roi_parser.add_argument('-w', '--width',
                        help="width of the cells border shown in the overlay",
                        default=2,
                        type=int)
    infer_roi_parser.add_argument('-c', '--color',
                        help="color of polygon shown on the overlay, refer to matplot lib colors for more information",
                        default="limegreen",
                        type=str)

    
    ########### TRAIN PARSER

    train_parser = subparsers.add_parser("train", help="Train model from pytable files of segmented nuclei")

    train_parser.add_argument('dataname',
                        help="dataset name, correspond to the pytables name under the following format: (dataname)_(phase).pytables",
                        type=str)
    train_parser.add_argument('-o', '--outdir',
                        help="output directory path for tensorboard and trained model",
                        default="./output/",
                        type=str)
    train_parser.add_argument('-p', '--dataset_path',
                        help="path to the directory that contains the pytables",
                        default='./',
                        type=str)
    train_parser.add_argument('-b', '--batch_size',
                        help="number of worker for dataloader",
                        default=5,
                        type=int)
    train_parser.add_argument('-n', '--n_worker',
                        help="number of worker for dataloader, default: min(batch_size,os.cpu_count())",
                        default=None,
                        type=int)
    train_parser.add_argument('-e', '--epoch',
                        help="number of epoch",
                        default=100,
                        type=int)
    train_parser.add_argument('-d', '--depth',
                        help="depth of the model",
                        default=3,
                        type=int)
    train_parser.add_argument('-w', '--width',
                        help="width of the model: define the number of filter in first layer (2**w) following with a exponential groth rate respective to the depth of the model.",
                        default=4,
                        type=int)
    
    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()
    if args.mode == "infer_wsi":
        main_wsi(args)

    else:
        raise ValueError("Please pick one of the following options: infer_wsi, infer_roi, train.")



if __name__ == "__main__":
    main()
