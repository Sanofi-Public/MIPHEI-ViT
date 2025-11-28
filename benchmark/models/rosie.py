from typing import Optional, Callable

import torch
import torch.nn as nn
import torchvision.models as models
import cv2


def get_model(num_outputs: Optional[int] = None, 
             use_context: bool = False, 
             use_mask: bool = False) -> nn.Module:
    """
    Creates and returns the model architecture.

    Args:
        num_outputs: Number of output features to predict
        use_context: Whether to use contextual features
        use_mask: Whether to use masking in the model

    Returns:
        PyTorch model instance
    """
    model = models.convnext_small(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model


def infer_sliding_window(x_input, model, P=128, S=8):
    bs, _, w, h = x_input.shape
    output = []
    with torch.inference_mode():
        for x in range(0, w - P + 1, S):
            for y in range(0, h - P + 1, S):
                x_input_crop = x_input[:, :, x:x+P, y:y+P]
                x_input_crop = torch.nn.functional.interpolate(x_input_crop, size=(224, 224), mode='bilinear')
                out_crop = model(x_input_crop)
                output.append(out_crop)

        w_out, h_out = (w - P) // S + 1, (h - P) // S + 1
        output = torch.stack(output, dim=2)
        output = output.view(bs, -1, w_out, h_out)
        output = (output.clamp(0., 1.) * 255).to(torch.uint8).cpu()
    return output

def retrieve_image_scale(pred, shape_crop, shape):
    pred_upscale = cv2.resize(pred[:shape_crop[0], :shape_crop[1]], None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
    #pred_upscale = pred_upscale[:256, :256]
    return pred_upscale
