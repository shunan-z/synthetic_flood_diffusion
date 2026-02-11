import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------------------------------------
# 1. Helper: Sinusoidal Time Embedding
# ---------------------------------------------------------
def sinusoidal_time_embedding(timesteps, dim):
    """
    Creates sinusoidal embeddings for timesteps (or noise levels).
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# ---------------------------------------------------------
# 2. Building Blocks (ResBlock, Down, Up)
# ---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Inject Time/Condition Embedding
        emb_out = self.emb_proj(F.silu(emb))  # (B, out_ch)
        h = h + emb_out[:, :, None, None]     # Broadcast to (B, out_ch, H, W)
        
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, emb_ch)
        self.res2 = ResBlock(out_ch, out_ch, emb_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, emb):
        h = self.res1(x, emb)
        h = self.res2(h, emb)
        skip = h
        h = self.down(h)
        return h, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(out_ch + skip_ch, out_ch, emb_ch)
        self.res2 = ResBlock(out_ch, out_ch, emb_ch)

    def forward(self, x, skip, emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # Concatenate skip connection
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        return x

# ---------------------------------------------------------
# 3. The UNet Architecture
# ---------------------------------------------------------
class SimpleUNetSmall(nn.Module):
    def __init__(self,
                 in_channels=4,    # [Noisy_Residual, XGB_Pred, Elev, Mask]
                 out_channels=1,   # [Predicted_Residual]
                 base_ch=32,
                 time_emb_dim=128,
                 cond_dim=5):      # [Cat, Speed, Tide, Dir_sin, Dir_cos]
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Global Condition MLP (e.g. Tide, Speed)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        emb_ch = time_emb_dim

        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Encoder (Down)
        self.down1 = DownBlock(base_ch, base_ch * 2, emb_ch)       # 64 -> 32
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, emb_ch)   # 32 -> 16
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, emb_ch)   # 16 -> 8

        # Bottleneck (Mid)
        self.mid1 = ResBlock(base_ch * 8, base_ch * 8, emb_ch)
        self.mid2 = ResBlock(base_ch * 8, base_ch * 8, emb_ch)

        # Decoder (Up)
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, emb_ch)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, emb_ch)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch, emb_ch)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_channels, 3, padding=1)

    def forward(self, x, t, global_cond):
        # 1. Embed Time
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # 2. Embed Global Conditions
        c_emb = self.cond_mlp(global_cond)
        
        # 3. Fuse Embeddings
        emb = t_emb + c_emb

        # 4. UNet Pass
        h = self.in_conv(x)

        h, s1 = self.down1(h, emb)
        h, s2 = self.down2(h, emb)
        h, s3 = self.down3(h, emb)

        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        h = self.up3(h, s3, emb)
        h = self.up2(h, s2, emb)
        h = self.up1(h, s1, emb)

        return self.out_conv(F.silu(self.out_norm(h)))

# ---------------------------------------------------------
# 4. EDM Configuration & Wrapper
# ---------------------------------------------------------
@dataclass
class EDMConfig:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2

    def sample_sigma(self, batch_size, device):
        """Sample noise levels for training"""
        log_sigma = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        sigma = log_sigma.exp()
        return sigma.clamp(self.sigma_min, self.sigma_max)

    def loss_weight(self, sigma):
        """Weighting to balance loss across noise levels"""
        sigma2 = sigma ** 2
        sigma_data2 = self.sigma_data ** 2
        return (sigma2 + sigma_data2) / (sigma * self.sigma_data) ** 2

class EDMResidual(nn.Module):
    def __init__(self, unet, cfg: EDMConfig, device="cpu"):
        super().__init__()
        self.unet = unet
        self.cfg = cfg
        self.device = device

    def get_preconditioning(self, sigma):
        # Karras preconditioning factors
        sigma_data = self.cfg.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out  = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in   = 1 / (sigma**2 + sigma_data**2).sqrt()
        return c_skip, c_out, c_in

    def denoise(self, r_t, sigma, sc_sane, miss_mask, global_cond):
        """
        Runs the UNet using EDM preconditioning.
        r_t: Noisy residual
        sc_sane: [Prediction, Elevation] (NaNs replaced with 0)
        miss_mask: 1 where data is missing, 0 otherwise
        """
        B = r_t.shape[0]
        sigma_broad = sigma.view(B, 1, 1, 1) if sigma.ndim < 4 else sigma

        c_skip, c_out, c_in = self.get_preconditioning(sigma_broad)
        c_noise = 0.25 * torch.log(sigma.flatten()) # Log-noise for time embedding

        # Stack inputs: [Scaled Noise, Pred, Elev, Mask] -> 4 Channels
        model_in = torch.cat([r_t * c_in, sc_sane, miss_mask], dim=1)

        # Network Prediction
        F_x = self.unet(model_in, c_noise, global_cond)

        # Skip connection formula
        return c_skip * r_t + c_out * F_x

    def p_losses(self, x0, spatial_cond, global_cond):
        """Calculates training loss for a batch"""
        x0 = x0.to(self.device)
        spatial_cond = spatial_cond.to(self.device)
        global_cond = global_cond.to(self.device)
        
        B = x0.shape[0]

        # 1. Prepare Target (True Residual)
        valid_mask = torch.isfinite(x0).float()
        miss_mask  = 1.0 - valid_mask
        
        x0_sane   = torch.nan_to_num(x0, nan=0.0)
        pred_sane = torch.nan_to_num(spatial_cond[:, 0:1], nan=0.0)
        sc_sane   = torch.nan_to_num(spatial_cond, nan=0.0)
        
        # We model the residual: Truth - XGB_Pred
        r0 = x0_sane - pred_sane

        # 2. Add Noise
        sigma = self.cfg.sample_sigma(B, self.device)
        sigma_broad = sigma.view(B, 1, 1, 1)
        eps = torch.randn_like(r0)
        r_t = r0 + sigma_broad * eps # Noisy residual

        # 3. Denoise
        r0_hat = self.denoise(r_t, sigma, sc_sane, miss_mask, global_cond)

        # 4. Loss (MSE on valid pixels only)
        mse = ((r0_hat - r0)**2) * valid_mask
        mse_per_sample = mse.sum(dim=(1,2,3)) / (valid_mask.sum(dim=(1,2,3)) + 1e-8)
        
        loss = (self.cfg.loss_weight(sigma) * mse_per_sample).mean()
        return loss

    @torch.no_grad()
    def sample(self, spatial_cond, global_cond, num_steps=32):
        """Generates samples using Euler method"""
        B, _, H, W = spatial_cond.shape
        device = self.device
        
        spatial_cond = spatial_cond.to(device)
        global_cond = global_cond.to(device)

        pred = spatial_cond[:, 0:1]
        sc_sane = torch.nan_to_num(spatial_cond, nan=0.0)
        miss_mask = (1.0 - torch.isfinite(pred).float())

        # 1. Start with Gaussian Noise
        r_t = torch.randn((B, 1, H, W), device=device) * self.cfg.sigma_max

        # 2. Setup Schedule
        rho = 7.0
        step_indices = torch.arange(num_steps, device=device)
        t_max = self.cfg.sigma_max**(1/rho)
        t_min = self.cfg.sigma_min**(1/rho)
        sigmas = (t_max + step_indices / (num_steps - 1) * (t_min - t_max))**rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)]) 

        # 3. Denoise Loop
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            sigma_batch = torch.full((B,), sigma, device=device)

            r0_hat = self.denoise(r_t, sigma_batch, sc_sane, miss_mask, global_cond)

            # Euler Step
            d_i = (r_t - r0_hat) / sigma
            r_t = r_t + (sigma_next - sigma) * d_i

        # 4. Recombine: Final = XGB_Pred + Predicted_Residual
        # (We only return values where input was valid)
        x_hat = (pred.nan_to_num(0.0) + r_t)
        
        return x_hat