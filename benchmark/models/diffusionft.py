import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel


def load_diffusion_pipeline(checkpoint_dir, device):
    noise_scheduler = DDPMScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")
    tokenizer    = CLIPTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", revision=None)
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", revision=None, variant=None).eval()
    vae          = AutoencoderKL.from_pretrained(checkpoint_dir, subfolder="vae", revision=None, variant=None).eval()
    unet         = UNet2DConditionModel.from_pretrained(checkpoint_dir, subfolder="unet", revision=None, variant=None).eval()
    generator = HE2MIFOneStepPipeline(
        vae=vae.to(device),
        unet=unet.to(device),
        scheduler=noise_scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer)
    generator = generator.to(device=device)
    return generator


class HE2MIFOneStepPipeline(DiffusionPipeline):
    """
    One-step (t = T-1, eps = 0) conditional pipeline.
    Uses:
      - Empty CLIP text encoding as encoder_hidden_states
      - Marker embeddings automatically generated from IDs

    Expects UNet that:
      - takes concatenation latent (rgb_lat + noise_lat) if in_channels == 8
      - else just noise_lat
      (matches your training)
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        # build & cache empty encoding

        # number of markers is inferred from class embedding size
        marker_embed_dim = unet.config.projection_class_embeddings_input_dim
        self.num_markers = marker_embed_dim // 2

        # build & cache marker embeddings
        self.empty_encoding = self.get_empty_encoding()
        self.marker_embeds = self.get_marker_embeds()

    @torch.no_grad()
    def get_empty_encoding(self):

        # standard empty prompt (""), same as SD
        empty_ids = self.tokenizer(
            [""], padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(self.text_encoder.device)

        empty_encoding = self.text_encoder(empty_ids, return_dict=False)[0]     # (1, 77, C)
        return empty_encoding.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def get_marker_embeds(self):

        ids = torch.arange(self.num_markers).to(torch.long)
        onehot = F.one_hot(ids, num_classes=self.num_markers).float()
        marker_embeds = torch.cat([torch.sin(onehot), torch.cos(onehot)], dim=-1)
        return marker_embeds.to(self.device, dtype=self.dtype)


    @torch.no_grad()
    def __call__(self, rgb, marker_ids=None, return_torch=False):

        device = self.device
        dtype = self.dtype
        B = rgb.shape[0]

        if marker_ids is None:
            marker_ids = list(range(self.num_markers))
        if isinstance(marker_ids, int):
            marker_ids = [marker_ids]

        M = len(marker_ids)

        # Encode RGB once
        rgb_lat = self.vae.encode(rgb).latent_dist.mode()
        rgb_lat = rgb_lat * self.vae.config.scaling_factor

        noise_lat = torch.zeros_like(rgb_lat)

        # Marker embeddings
        marker_embeds = self.marker_embeds[marker_ids]  # (M, D)
        marker_embeds = marker_embeds.unsqueeze(1).expand(-1, B, -1)  
        marker_embeds = marker_embeds.reshape(M*B, -1)  # (MB, D)

        # Duplicate latents for each marker
        rgb_lat = rgb_lat.repeat(M, 1, 1, 1)
        noise_lat = noise_lat.repeat(M, 1, 1, 1)

        # Empty prompt encoding
        enc = self.empty_encoding.expand(M*B, -1, -1)

        # timestep
        T = self.scheduler.config.num_train_timesteps
        t = torch.full((M*B,), T-1, device=device, dtype=torch.long)

        # UNet input
        if self.unet.config.in_channels == (rgb_lat.shape[1] + noise_lat.shape[1]):
            unet_in = torch.cat([rgb_lat, noise_lat], dim=1)
        else:
            unet_in = noise_lat

        # UNet (batched)
        model_out = self.unet(
            unet_in,
            t,
            encoder_hidden_states=enc,
            class_labels=marker_embeds,
            return_dict=False
        )[0]

        # Convert to x0 (latent) — still batched
        alphas = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        alpha_t = alphas[t].view(-1,1,1,1)
        beta_t = (1 - alphas[t]).view(-1,1,1,1)

        if self.scheduler.config.prediction_type == "v_prediction":
            x0_lat = alpha_t.sqrt() * noise_lat - beta_t.sqrt() * model_out
        elif self.scheduler.config.prediction_type == "epsilon":
            x0_lat = (noise_lat - beta_t.sqrt() * model_out) / alpha_t.sqrt()
        else:
            x0_lat = model_out

        x0_lat = x0_lat / self.vae.config.scaling_factor  # (MB,C,H,W)

        # VAE decode per marker to avoid OOM
        outs = []
        x0_lat = x0_lat.reshape(M, B, *x0_lat.shape[1:])  # (M,B,C,H,W)

        for m in range(M):
            pred = self.vae.decode(x0_lat[m]).sample  # (B,1,H,W)
            pred = pred.mean(dim=1)  # (B,1,H,W)
            outs.append(pred)

        outs = torch.stack(outs, dim=1)  # (B,M,1,H,W)
        outs = outs.clamp(-1,1)
        outs = (outs / 2 + 0.5).clamp(0, 1)
        if return_torch:
            return outs
        return outs.permute(0, 2, 3, 1).cpu().numpy()
