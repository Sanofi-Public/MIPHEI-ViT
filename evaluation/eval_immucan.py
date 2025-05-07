import pyvips
from omegaconf import OmegaConf
import json
import pandas as pd
from pathlib import Path
import torch
import argparse
from tqdm import tqdm
from tqdm import tqdm
from timm.layers import resample_abs_pos_embed
from timm.layers import resample_patch_embed, resize_rel_pos_bias_table
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from src.dataset import get_width_height, NormalizationLayer, get_effective_width_height,\
        TileImg2ImgSlideDataset, get_input_mean_std
from src.generators import get_generator
from src.metrics import CellMetrics


def validate_load_info(load_info):
    """
    Validates the result of model.load_state_dict(..., strict=False).

    Raises:
        ValueError if unexpected keys are found,
        or if missing keys are not related to the allowed encoder modules.
    """
    # 1. Raise if any unexpected keys
    if load_info.unexpected_keys:
        raise ValueError(f"Unexpected keys in state_dict: {load_info.unexpected_keys}")

    # 2. Raise if any missing keys are not part of allowed encoder modules
    for key in load_info.missing_keys:
        if ".lora" in key:
            raise ValueError(f"Missing LoRA checkpoint in state_dict: {key}")
        elif not any(part in key for part in ["encoder.vit.", "encoder.model."]):
            raise ValueError(f"Missing key in state_dict: {key}")


def resize_embed_hemit_statedict(state_dict, model):
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'attn_mask')]):
                continue

        if 'swinT.patch_embed.proj.weight' in k:
            _, _, H, W = model.swinT.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )
        new_state_dict[k] = v

    return new_state_dict


def get_generator_state_dict(state_dict):
    generator_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            generator_state_dict[k.replace("generator.", "")] = v
    return generator_state_dict


def correlation_analysis(tile_sums, marker_name, figpath):
    corr, _ = pearsonr(tile_sums[f"{marker_name}_count"], tile_sums[f"{marker_name}_pos_logreg"])
    formatted_corr = f"{corr:.3f}" if abs(corr) >= 0.01 else f"{corr:.2e}"  # Use scientific notation if small

    sns.regplot(x=tile_sums[f"{marker_name}_count"], y=tile_sums[f"{marker_name}_pos_logreg"],
                line_kws={'color': 'black'}, color=COLORS[marker_name], ci=None)
    plt.text(0.05, 0.95, f"Pearson r = {formatted_corr}", transform=plt.gca().transAxes, 
            fontsize=20, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    # Better title without correlation
    plt.title(f"{marker_name}", fontsize=32)
    plt.xlabel('Target', fontsize=14)
    plt.ylabel('Pred', fontsize=14)
    plt.savefig(figpath, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    return corr

COLORS = {"CD3e": "orange",
          "CD8a": "green",
          "CD4": "blue",
          "FOXP3": "purple",
          "Pan-CK": "red"}
DATASET_CONFIG_PATH = "../configs/data/immucan.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint_dir')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    config_path = str(Path(checkpoint_dir) / "config.yaml")

    cfg = OmegaConf.load(config_path)
    cfg_data = OmegaConf.load(DATASET_CONFIG_PATH)
    for key in ["slide_dataframe_path", "train_dataframe_path", "val_dataframe_path", "test_dataframe_path", "channel_stats_path"]:
        if key in cfg_data.data:
            cfg.data[key] = cfg_data.data[key]

    dataframe = pd.read_csv(cfg.data.test_dataframe_path)

    # trick to make faster (but can be optimized)
    dataframe["target_path"] = dataframe["image_path"]
    # trick to make cell metric compatible with our cell count analysis
    slide_dataframe = pd.DataFrame()
    slide_dataframe["in_slide_name"] = dataframe["image_path"].apply(lambda x: Path(x).stem).tolist()
    slide_dataframe["nuclei_csv_path"] = None

    with open(Path("..") / cfg.data.channel_stats_path, "r") as f:
        channel_stats = json.load(f)

    width, height = get_width_height(dataframe)
    width, height = get_effective_width_height(width, height, train=True)

    nc_out = len(cfg.data.targ_channel_names)
    nc_in = 3
    print("{} width / {} height".format(width, height))
    print("{} inputs channels / {} output channels".format(nc_in, nc_out))


    channel_stats_rgb = get_input_mean_std(cfg, channel_stats["RGB"])
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

    torch.cuda.empty_cache()

    generator = get_generator(cfg.model.model_name, width, nc_in, nc_out, cfg)
    use_safetensors = (Path(checkpoint_dir) / "model.safetensors").exists()
    if use_safetensors:
        from safetensors.torch import load_file
        checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
        state_dict = load_file(checkpoint_path, device="cpu")
        strict_load = False
        print("Loading checkpoint from safetensors")
    else:
        checkpoint_path = str(Path(checkpoint_dir) / "model.weights.ckpt")
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = get_generator_state_dict(state_dict)
        strict_load = True
        print("Loading checkpoint from ckpt")
    if hasattr(generator, "swinT"):
        state_dict = resize_embed_hemit_statedict(state_dict, generator)
    
    load_info = generator.load_state_dict(state_dict, strict=strict_load)
    if use_safetensors:
        validate_load_info(load_info)
    generator = generator.eval().cuda().half()

    cell_metrics = CellMetrics(slide_dataframe, marker_names=cfg.data.targ_channel_names,
                               min_area=20).cuda()
    n_marker = len(cell_metrics.marker_cols)
    logreg = torch.nn.Linear(n_marker, n_marker)
    logreg_state_dict = torch.load(str(Path(checkpoint_dir) / "logreg.pth"), map_location="cpu")
    logreg.load_state_dict(logreg_state_dict)
    logreg.eval()

    dataset = TileImg2ImgSlideDataset(
            dataframe=dataframe, preprocess_input_fn=preprocess_input_fn,
            spatial_augmentations=None, return_nuclei=True)

    num_workers = 6
    batch_size = 4
    device = "cpu"
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=device!="cpu",
        shuffle=False, drop_last=False, num_workers=num_workers
    )

    for batch in tqdm(dataloader):
        x = batch["image"].cuda()
        nuclei_masks = batch["nuclei"].cuda()
        tile_names = batch["tile_name"]
        with torch.inference_mode():
            out = generator(x.half()).float()
        cell_metrics.update(out, nuclei_masks, tile_names)

    cell_dataframe = cell_metrics.get_dataframe_cell_pred()
    cell_dataframe = cell_dataframe.rename(columns={"slide_name": "tile_name"})
    cell_metrics.reset()

    with torch.inference_mode():
        cell_probs = torch.sigmoid(logreg(torch.from_numpy(
            cell_dataframe[cell_metrics.marker_pred_cols].values).float())).numpy()
        cell_preds = cell_probs > 0.5

    pred_columns = [f"{col}_logreg" for col in cell_metrics.marker_cols]
    cell_dataframe[pred_columns] = cell_preds

    tile_sums = cell_dataframe.groupby("tile_name")[pred_columns].sum().reset_index(drop=False)
    dataframe["tile_name"] = dataframe["image_path"].apply(lambda x: Path(x).stem)
    dataframe = dataframe.drop(columns=["image_path", "nuclei_path"])
    dataframe = dataframe.rename(columns={"CD3_count": "CD3e_count", "CD8_count": "CD8a_count"})
    tile_sums = tile_sums.merge(dataframe, on="tile_name")

    corr_results = []
    for marker_name in COLORS.keys():
        figpath = str(Path(checkpoint_path).parent / f"{marker_name}_corr.png")
        corr = correlation_analysis(tile_sums, marker_name, figpath)
        corr_results.append([marker_name, corr])
    corr_results_df = pd.DataFrame(columns=["Marker", "Pearson"], data=corr_results)
    corr_results_df.to_csv(str(Path(checkpoint_path).parent / "immucan_corr.csv"), index=False)
    tile_sums.to_csv(str(Path(checkpoint_path).parent / "immucan_tile_sums.csv"), index=False)
