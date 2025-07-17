"""
Implementation of common histology-specific foundation models using timm.

This module is self-contained and can be reused in other packages as-is.
Weights are loaded from Hugging Face; ensure you have access to the models.
All implementation where tested to match original models outputs.
"""

import torch
import torch.nn as nn

from safetensors.torch import load_file

import timm
from timm.layers import resample_abs_pos_embed
from timm.layers.helpers import to_2tuple
from timm.models import VisionTransformer, SwinTransformer, ResNet
from timm.models import load_state_dict_from_hf, parse_model_name


FOUNDATION_HF_CKPT_REGISTRY = {
    "univ2": "hf-hub:MahmoodLab/UNI2-h",
    "hoptimus0": "hf-hub:bioptimus/H-optimus-0",
    "provgigapath": "hf_hub:prov-gigapath/prov-gigapath",
    "sp85m": "hf_hub:MountSinaiCompPath/SP85M",
    "phikonv2": "hf_hub:owkin/phikon-v2",
    "restnet50_lunit_swav": "hf_hub:1aurent/resnet50.lunit_swav",
    "ctranspath": "hf_hub:jamesdolezal/CTransPath"
}


def univ2(
    img_size: int | tuple[int, int],
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_path_rate: float = 0.,
    global_pool: str = ""
) -> VisionTransformer:
    """
    Create and return UNIv2 foundation model as described by Mahmood Lab AI for Pathology \
        @ Harvard/BWH.

    It is a ViT-G/14 trained on 224x224pxs tiles using DINOv2.
    This function constructs the UNIv2 foundation model optionally loading pretrained weights from
    a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_path_rate (float, optional): Stochastic depth rate for regularization. Defaults to 0.0.
        global_pool (str, optional): Type of global pooling to use.
            Defaults to "" (no global pooling).

    Returns:
        VisionTransformer: An instance of the Univ2 Vision Transformer model, optionally loaded
            with pretrained weights.

    References:
        - Paper: https://www.nature.com/articles/s41591-024-02857-3
        - HuggingFace Model: https://huggingface.co/MahmoodLab/UNI2-h
        - Code: https://github.com/mahmoodlab/UNI
    """
    model = timm.create_model(
        model_name='vit_giant_patch14_224', img_size=img_size,
        patch_size=14, depth=24, num_heads=24,
        drop_path_rate=drop_path_rate, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667*2,
        num_classes=0,  no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8,
        global_pool=global_pool, pretrained=False,
        dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["univ2"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for UNIv2 foundation model")
    return model


def hoptimus0(
    img_size: int | tuple[int, int],
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_path_rate: float = 0.,
    global_pool: str = ""
) -> VisionTransformer:
    """
    Create and return H-Optimus-0 foundation model from Bioptimus.

    It is a ViT-H/14 trained on 224x224pxs tiles using DINOv2 and IBot.
    This function constructs the H-Optimus-0 foundation model optionally loading pretrained weights
    from a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_path_rate (float, optional): Stochastic depth rate for regularization. Defaults to 0.0.
        global_pool (str, optional): Type of global pooling to use.
            Defaults to "" (no global pooling).

    Returns:
        VisionTransformer: An instance of the H-Optimus-0 Vision Transformer model, optionally
            loaded with pretrained weights.

    References:
        - HuggingFace Model: https://huggingface.co/bioptimus/H-optimus-0
        - Code: https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0
    """
    model = timm.create_model(
        "vit_giant_patch14_reg4_dinov2", img_size=img_size,
        drop_path_rate=drop_path_rate, num_classes=0,
        global_pool=global_pool, pretrained=False, init_values=1e-5,
        dynamic_img_size=False)

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["hoptimus0"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for hoptimus0 foundation model")
    return model


def sp85m(
    img_size: int | tuple[int, int],
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_path_rate: float = 0.,
    global_pool: str = ""
) -> VisionTransformer:
    """
    Create and return SPM85M foundation model from Mount Sinai Health System.

    It is a ViT-B/16 trained on 224x224pxs tiles (423k WSIs).
    This function constructs the SPM85M foundation model optionally loading pretrained weights from
    a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_path_rate (float, optional): Stochastic depth rate for regularization. Defaults to 0.0.
        global_pool (str, optional): Type of global pooling to use.
            Defaults to "" (no global pooling).

    Returns:
        VisionTransformer: An instance of the H-Optimus-0 Vision Transformer model, optionally
            loaded with pretrained weights.

    References:
        - HuggingFace Model: https://huggingface.co/MountSinaiCompPath/SP85M
    """
    model = timm.create_model(
        "vit_base_patch16_224", img_size=img_size,
        num_classes=0, drop_path_rate=drop_path_rate,
        global_pool=global_pool, pretrained=False,
        dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["sp85m"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size=img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for sp85m foundation model")
    return model


def provgigapath(
    img_size: int | tuple[int, int],
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_path_rate: float = 0.,
    global_pool: str = ""
) -> VisionTransformer:
    """Create and return ProvGigapath foundation model from Microsoft Research.

    It is a ViT-G/16 trained on 224x224pxs tiles using DINOv2.
    This function constructs the ProvGigapath foundation model optionally loading pretrained
    weights from a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_path_rate (float, optional): Stochastic depth rate for regularization. Defaults to 0.0.
        global_pool (str, optional): Type of global pooling to use.
            Defaults to "" (no global pooling).

    Returns:
        VisionTransformer: An instance of the H-Optimus-0 Vision Transformer model, optionally
            loaded with pretrained weights.

    References:
        - Paper: https://www.nature.com/articles/s41586-024-07441-w
        - HuggingFace Model: https://huggingface.co/prov-gigapath/prov-gigapath
        - Code: https://github.com/prov-gigapath/prov-gigapath
    """
    model = timm.create_model(
        "vit_giant_patch14_dinov2", img_size=img_size,
        num_classes=0, patch_size=16, global_pool=global_pool,
        drop_path_rate=drop_path_rate, pretrained=False,
        init_values=1e-5, dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["provgigapath"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size=img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for provgigapath foundation model")
    return model


def phikonv2(
    img_size: int | tuple[int, int],
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_path_rate: float = 0.,
    global_pool: str = ""
) -> VisionTransformer:
    """
    Create and return Phikon-v2 foundation model from Owkin.

    It is a ViT-L/16 trained on 224x224pxs tiles using DINOv2.
    This function constructs the Phikon-v2 foundation model optionally loading pretrained weights
    from a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_path_rate (float, optional): Stochastic depth rate for regularization. Defaults to 0.0.
        global_pool (str, optional): Type of global pooling to use.
            Defaults to "" (no global pooling).

    Returns:
        VisionTransformer: An instance of the H-Optimus-0 Vision Transformer model, optionally
            loaded with pretrained weights.

    References:
        - Paper: https://arxiv.org/abs/2409.09173
        - HuggingFace Model: https://huggingface.co/owkin/phikon-v2
        - Code: https://github.com/owkin/HistoSSLscaling
    """
    model = timm.create_model(
        "vit_large_patch14_dinov2", img_size=img_size,
        num_classes=0, patch_size=16, global_pool=global_pool,
        drop_path_rate=drop_path_rate, pretrained=False,
        dynamic_img_size=False)

    if pretrained or ckpt_path:
        depth = len(model.blocks)
        embed_dim = model.embed_dim
        if ckpt_path:
            state_dict = load_file(ckpt_path)
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["phikonv2"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = hf2timm_checkpoint_conversion(state_dict, depth, embed_dim)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size=img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for phikonv2 foundation model")
    return model


def restnet50_lunit_swav(
    img_size: int | tuple[int, int] | None = None,
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_rate: float = 0.,
) -> ResNet:
    """
    Create and return ResNet50Lunit foundation model from Lunit.

    It is a ResNet50 model trained using SwAV.
    This function constructs the ResNet50Lunit foundation model optionally loading pretrained
    weights from a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_rate (float, optional): Dropout rate for regularization. Defaults to 0.0.

    Returns:
        ResNet50: An instance of the ResNet50Lunit model, optionally loaded with pretrained weights.

    References:
        - Paper: https://arxiv.org/pdf/2212.04690
        - HuggingFace Model: https://huggingface.co/1aurent/resnet50.lunit_swav
        - Code: https://github.com/lunit-io/benchmark-ssl-pathology
    """
    model = timm.create_model(
        model_name="resnet50",
        num_classes=0,
        drop_rate=drop_rate,
        pretrained=True,
        )

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = load_file(ckpt_path)
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["restnet50_lunit_swav"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for restnet50_lunit_swav foundation model")
    return model


def ctranspath(
    img_size: int | tuple[int, int],
    pretrained: bool = True,
    ckpt_path: str | None = None,
    drop_path_rate: float = 0.,
    global_pool: str = ""
) -> SwinTransformer:
    """
    Create and return CTransPath foundation model from Sichuan University.

    It is a custom Swin Transformer trained using custom MocoV3.
    This function constructs the CTransPath foundation model optionally loading pretrained weights
    from a checkpoint or HuggingFace.

    Args:
        img_size (int or tuple): The input image size (height, width) for the model.
        pretrained (bool, optional): If True, loads pretrained weights from HuggingFace or a
            specified checkpoint. Defaults to True.
        ckpt_path (str, optional): Path to a local checkpoint file to load weights from. If None,
            loads from HuggingFace. Defaults to None.
        drop_path_rate (float, optional): Stochastic depth rate for regularization. Defaults to 0.0.
        global_pool (str, optional): Type of global pooling to use.
            Defaults to "" (no global pooling).

    Returns:
        SwinTransformer: An instance of the CTransPath model, optionally loaded with pretrained
            weights.

    References:
        - Paper: https://www.sciencedirect.com/science/article/pii/S1361841522002043
        - HuggingFace Model: https://huggingface.co/jamesdolezal/CTransPath
        - Code: https://github.com/Xiyue-Wang/TransPath
    """
    model = timm.create_model(
        model_name="swin_tiny_patch4_window7_224",
        img_size=img_size,
        num_classes=0,
        embed_layer=ConvStem,  # defined below
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
        pretrained=False,
        )

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path)["model"]
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["ctranspath"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True,
                                                 filename="ctranspath.pth")["model"]
        state_dict = adapt_checkpoint_ctranspath(state_dict)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("Warning: random initialization for ctranspath foundation model")
    return model


FOUNDATION_MODEL_REGISTRY = {
    "univ2": univ2,
    "hoptimus0": hoptimus0,
    "sp85m": sp85m,
    "provgigapath": provgigapath,
    "phikonv2": phikonv2,
    "restnet50_lunit_swav": restnet50_lunit_swav,
    "ctranspath": ctranspath
}


def resize_pos_embed_statedict(state_dict, model: nn.Module, img_size: int):
    """Resize the position embedding in the state_dict if the input image size differs from 224."""
    if img_size != 224:
        old_pos_embed = state_dict['pos_embed']
        pos_embed = resample_abs_pos_embed(
            old_pos_embed,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=0 if model.no_embed_class else model.num_prefix_tokens,
        )
        state_dict['pos_embed'] = pos_embed
    return state_dict


def hf2timm_checkpoint_conversion(state_dict, depth: int, embed_dim: int,
                                  use_swiglu_ffn: bool = False):
    """Convert a Hugging Face DINOv2 checkpoint to a timm-compatible state_dict.

    Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/convert_dinov2_to_hf.py#L146
    """
    rename_keys = create_rename_keys(depth=depth, use_swiglu_ffn=use_swiglu_ffn)
    for dest, src in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = convert_hf_qkv_to_timm(state_dict, depth, embed_dim)

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "weights_in" in key:
            key = key.replace("weights_in", "w12")
        if "weights_out" in key:
            key = key.replace("weights_out", "w3")
        state_dict[key] = val

    state_dict.pop("mask_token")
    return state_dict


def create_rename_keys(depth: int, use_swiglu_ffn: bool):
    """Return key renaming rules for state_dict conversion."""
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("mask_token", "embeddings.mask_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))

    for i in range(depth):
        # layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))
        # MLP
        if use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))
        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))
        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight",
                            f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias",
                            f"encoder.layer.{i}.attention.output.dense.bias"))

    # final layernorm
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    return rename_keys


def rename_key(dct: dict, old: str, new: str) -> None:
    """Rename a key in a dictionary from 'old' to 'new'."""
    val = dct.pop(old)
    dct[new] = val


def convert_hf_qkv_to_timm(state_dict, depth: int, embed_dim: int):
    """Convert HuggingFace QKV weights to TIMM format for Vision Transformer models."""
    for i in range(depth):
        # Extract query, key, and value weights
        query_weight = state_dict.pop(f"encoder.layer.{i}.attention.attention.query.weight")
        key_weight = state_dict.pop(f"encoder.layer.{i}.attention.attention.key.weight")
        value_weight = state_dict.pop(f"encoder.layer.{i}.attention.attention.value.weight")

        # Concatenate weights into TIMM's format
        in_proj_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)

        # Extract query, key, and value biases
        query_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.query.bias")
        key_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.key.bias")
        value_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.value.bias")

        # Concatenate biases into TIMM's format
        in_proj_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)

        # Store in TIMM format
        state_dict[f"blocks.{i}.attn.qkv.weight"] = in_proj_weight
        state_dict[f"blocks.{i}.attn.qkv.bias"] = in_proj_bias

    return state_dict


class ConvStem(nn.Module):
    """Custom Patch Embed Layer used in CTransPath.

    Adapted from https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py#L6-L44
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 768, norm_layer: bool = None, **kwargs):
        super().__init__()

        # Check input constraints
        assert patch_size == 4, "Patch size must be 4"
        assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Create stem network
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1,
                                  bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        # Apply normalization layer (if provided)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for processing input tensor through projection and normalization layers."""
        B, C, H, W = x.shape

        # Check input image size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model \
                ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x


def adapt_checkpoint_ctranspath(state_dict) -> dict:
    """
    Adapts CTransPath checkpoint keys to match updated timm layer naming conventions.

    This function updates layer names in the checkpoint state_dict to be compatible with
    recent timm versions, which changed the indexing of downsample layers. It also skips
    keys related to relative position indices and attention masks (no change in weights).

    Returns:
        dict: A new state_dict with updated keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if ".downsample.norm" in k or "downsample.reduction" in k:
            k_split = k.split(".")
            k_split[1] = str(int(k_split[1]) + 1)
            new_k = ".".join(k_split)
        elif 'relative_position_index' in k or 'attn_mask' in k:
            continue
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict
