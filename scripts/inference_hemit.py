import argparse
import torch
import json
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append("../")
from src.dataset import Img2ImgSlideDataset, NormalizationLayer, dataloader_worker_init_fn
from src.unet import UnetMultiHeads


def get_model():
    model = UnetMultiHeads(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        decoder_use_batchnorm=True,
        dropout=0.,
        in_channels=3,
        classes=1,
        activation=torch.nn.Tanh)
    return model


def extract_generator_state_dict(state_dict):
    generator_state_dict = {}
    for var_name, var in state_dict["state_dict"].items():
        if var_name.startswith("generator"):
            new_var_name = var_name.replace("generator.", "")
            generator_state_dict[new_var_name] = var
    return generator_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    image_output_dir = Path(output_dir) / "images"
    image_output_dir.mkdir(exist_ok=True)
    with open("../channel_stats.json", "r") as f:
        channel_stats = json.load(f)

    preprocess_input_fn = NormalizationLayer(channel_stats["RGB"], mode="he")

    num_workers = os.cpu_count()
    targ_channel_idxs = [2]
    test_dataset = Img2ImgSlideDataset(
        str(Path(dataset_dir) / "test"),
        targ_channel_idxs=targ_channel_idxs,
        preprocess_input_fn=preprocess_input_fn,
        preprocess_target_fn=None)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=num_workers, worker_init_fn=dataloader_worker_init_fn)
    
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model()
    state_dict = torch.load(checkpoint_path)
    state_dict = extract_generator_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    idx_image = 0
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        x, target = batch
        with torch.no_grad():
            out = model(x.to(device))
            out = torch.clip((out + 0.9) * 255 / 1.8, 0., 255).permute((0, 2, 3, 1))
            out = out.to(torch.uint8).cpu()
        out = out.numpy()
        image = np.uint8(preprocess_input_fn.unormalize(x.permute((0, 2, 3, 1)).numpy()))
        target = target.permute((0, 2, 3, 1)).numpy()
        for idx in range(len(image)):
            image_curr = image[idx]
            out_curr = out[idx]
            target_curr = target[idx]
            cv2.imwrite(str(image_output_dir / f"{idx_image}_real_A.png"), image_curr)
            cv2.imwrite(str(image_output_dir / f"{idx_image}_fake_B.png"), out_curr)
            cv2.imwrite(str(image_output_dir / f"{idx_image}_real_B.png"), target_curr)

            idx_image += 1
