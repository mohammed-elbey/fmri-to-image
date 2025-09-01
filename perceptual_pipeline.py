import torch
import torch.nn as nn
from diffusers import AutoencoderKL, StableDiffusionPipeline

class LowLevelManipulationNetwork(nn.Module):
    """
    Transforms fMRI embeddings into feature maps that manipulate Stable Diffusion's U-Net 
    for spatial/perceptual control during image generation.
    """
    def __init__(self, 
                 sd_vae_path: str = "stabilityai/sd-vae-ft-mse",
                 sd_unet_path: str = "stabilityai/stable-diffusion-2-1",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Load pretrained VAE for encoding images to latent space
        self.vae = AutoencoderKL.from_pretrained(sd_vae_path).to(device)
        self.vae.eval() 

        # Load frozen SD U-Net for feature map manipulation
        pipe = StableDiffusionPipeline.from_pretrained(sd_unet_path).to(device)
        self.sd_unet = pipe.unet
        self.sd_unet.requires_grad_(False)
        
        # Transform fMRI spatial data [B,1,256,256] -> U-Net feature maps [B,320,64,64]
        self.fmri_to_features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),      # [B,1,256,256] -> [B,64,256,256]
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # -> [B,128,256,256]
            nn.GELU(),
            nn.Conv2d(128, 320, kernel_size=3, padding=1),   # -> [B,320,256,256]
            nn.GELU(),
            nn.AdaptiveAvgPool2d((64, 64))                   # -> [B,320,64,64]
        ).to(self.device)
        
        # Project semantic embeddings [B,512] -> [B,1024] for U-Net conditioning
        self.semantic_proj = nn.Linear(512, 1024).to(self.device)
        # Project fMRI features [B,320] -> [B,512] for semantic alignment
        self.fmri_proj = nn.Linear(320, 512).to(self.device)

        # U-Net block channel dimensions for each down/up block
        self.unet_channels = [320, 640, 1280, 1280, 1280, 1280, 640, 320]
        
        # Zero-initialized convolutions for residual fMRI injection into each U-Net block
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(320, out_channels, kernel_size=1, padding=0)
            for out_channels in self.unet_channels
        ])

        # Initialize all zero convolutions to zeros for stable training start
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.zero_convs.to(self.device)
        
    def forward_with_fmri(self, noisy_latents, timestep, encoder_hidden_states, fmri_features, control_scale):
        """Forward pass through U-Net with fMRI feature injection via hooks"""
        
        def hook_fn(module, input, output, i=[0]):
            # Get corresponding fMRI feature map for this U-Net block
            fmap = self.zero_convs[i[0]](fmri_features)  # [B,320,64,64] -> [B,block_channels,64,64]

            # Handle both tuple and tensor outputs from U-Net blocks
            if isinstance(output, tuple):
                out0 = output[0]
                # Resize fMRI features to match block output spatial dimensions
                if fmap.shape[-2:] != out0.shape[-2:]:
                    fmap = torch.nn.functional.interpolate(fmap, size=out0.shape[-2:], mode='bilinear', align_corners=False)
                # Inject scaled fMRI features as residual
                out0 = out0 + fmap * control_scale
                i[0] += 1
                return (out0,) + output[1:]
            else:
                if fmap.shape[-2:] != output.shape[-2:]:
                    fmap = torch.nn.functional.interpolate(fmap, size=output.shape[-2:], mode='bilinear', align_corners=False)
                output = output + fmap * control_scale
                i[0] += 1
                return output

        # Register hooks on all down and up blocks
        hooks = []
        for block in self.sd_unet.down_blocks + self.sd_unet.up_blocks:
            hooks.append(block.register_forward_hook(hook_fn))

        # Forward pass with fMRI feature injection
        with torch.no_grad():
            unet_output = self.sd_unet(
                noisy_latents,      # [B,4,64,64] noisy latents
                timestep,           # [B] timesteps
                encoder_hidden_states=encoder_hidden_states  # [B,77,1024] text conditioning
            ).sample

        # Clean up hooks
        for h in hooks:
            h.remove()

        return unet_output  # [B,4,64,64] predicted noise

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode RGB image [B,3,H,W] to VAE latent space [B,4,64,64]"""
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample() * 0.18215
        return latent

    def transform_fmri(self, fmri_embedding: torch.Tensor) -> torch.Tensor:
        """Transform fMRI spatial data [B,256,256] to U-Net features [B,320,64,64]"""
        fmri_embedding = fmri_embedding.unsqueeze(1)  # [B,256,256] -> [B,1,256,256]
        return self.fmri_to_features(fmri_embedding)  # -> [B,320,64,64]
    
    def project_fmri_to_semantic(self, fmri_embedding: torch.Tensor) -> torch.Tensor:
        """Project fMRI features [B,320,64,64] to semantic space [B,512] for alignment loss"""
        if fmri_embedding.dim() == 3:
            fmri_embedding = fmri_embedding.unsqueeze(1)

        # Global average pooling: [B,320,64,64] -> [B,320]
        pooled = fmri_embedding.mean(dim=[2, 3])
        return self.fmri_proj(pooled)  # [B,320] -> [B,512]

    def manipulate_unet(
        self, 
        noisy_latents: torch.Tensor,      # [B,4,64,64] noisy latents
        fmri_features: torch.Tensor,      # [B,320,64,64] fMRI features
        timestep: torch.Tensor,           # [B] timesteps
        hidden_state: torch.Tensor,       # [B,512] semantic embeddings
        control_scale: float = 1.0        # fMRI influence strength
    ) -> torch.Tensor:
        """Manipulate U-Net forward pass with fMRI conditioning"""
        
        # Project semantic embeddings and expand for text conditioning format
        hidden_state = self.semantic_proj(hidden_state)  # [B,512] -> [B,1024]
        hidden_state = hidden_state.unsqueeze(1).repeat(1, 77, 1)  # -> [B,77,1024]
        
        return self.forward_with_fmri(noisy_latents, timestep, hidden_state, fmri_features, control_scale)

if __name__ == "__main__":
    llmn = LowLevelManipulationNetwork(device="cuda")
    
    # Example usage with proper dimensions
    dummy_fmri = torch.randn(2, 256, 256).to("cuda")        # fMRI spatial data
    dummy_image = torch.randn(2, 3, 512, 512).to("cuda")    # RGB images
    dummy_semantic = torch.randn(2, 512).to("cuda")         # CLIP embeddings
    
    # Encode image to latent space
    latent = llmn.encode_image(dummy_image)                  # -> [2,4,64,64]
    
    # Transform fMRI to feature maps
    fmri_features = llmn.transform_fmri(dummy_fmri)         # -> [2,320,64,64]
    
    # Add noise for diffusion process
    noisy_latent = torch.randn_like(latent)                 # [2,4,64,64]
    timestep = torch.tensor([500, 300]).to("cuda")         # [2] timesteps
    
    # Generate with fMRI conditioning
    output = llmn.manipulate_unet(
        noisy_latent, fmri_features, timestep, dummy_semantic, control_scale=0.5
    )  # -> [2,4,64,64] predicted noise