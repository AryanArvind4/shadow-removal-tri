"""
Stage 1 — Diffusion Model Architecture
U-Net with time embedding, designed for grayscale PCB X-ray images (1 channel).
This is the score network sθ(x(t), t) that SDEdit needs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time (noise level) embedding ─────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Encodes the time step t as a sinusoidal vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── Building blocks ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection — match channels if needed
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing global structure."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


# ── Full U-Net ────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Score network for grayscale PCB X-ray images.
    Input  : noisy image x(t)  [B, 1, H, W] + time step t [B]
    Output : predicted score   [B, 1, H, W]

    Architecture:
        Encoder: 4 resolution levels, each with 2 ResBlocks + optional attention
        Bottleneck: 2 ResBlocks + attention
        Decoder: mirrors encoder with skip connections
    """

    def __init__(
        self,
        image_channels=1,          # grayscale X-ray
        base_channels=64,          # channel width at first level
        channel_mults=(1, 2, 4, 8),# multiplier per resolution level
        attn_resolutions=(16,),    # apply attention at these spatial sizes
        num_res_blocks=2,
        dropout=0.1,
        image_size=256
    ):
        super().__init__()

        self._channel_mults = channel_mults
        self._num_res_blocks = num_res_blocks

        time_dim = base_channels * 4

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial projection
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        resolution = image_size
        skip_channels = [ch]   # track channels for skip connections

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                if resolution in attn_resolutions:
                    self.down_blocks.append(AttentionBlock(ch))
                skip_channels.append(ch)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(ch))
                resolution //= 2
                skip_channels.append(ch)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_block1  = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn    = AttentionBlock(ch)
        self.mid_block2  = ResBlock(ch, ch, time_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up_blocks   = nn.ModuleList()
        self.upsamplers  = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_dim, dropout)
                )
                ch = out_ch
                if resolution in attn_resolutions:
                    self.up_blocks.append(AttentionBlock(ch))
            if level > 0:
                self.upsamplers.append(Upsample(ch))
                resolution *= 2

        # Final output projection
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, image_channels, 3, padding=1)

    def forward(self, x, t):
        """
        x : noisy image  [B, 1, H, W]  in [-1, 1]
        t : time steps   [B]            in [0, 1]
        Returns predicted score [B, 1, H, W]

        Skip connection layout (pushed in order, popped LIFO by decoder):
          [init_conv] + per-level [RB0, RB1, DS] + last-level [RB0, RB1]
          = 1 + 3*(n_levels-1) + 2 = 12 items for default 4-level config
        """
        t_emb = self.time_mlp(t)

        h = self.init_conv(x)
        skips = [h]

        # ── Encoder: ResBlocks (+ optional Attention) interleaved with Downsamples
        block_idx = 0
        n_levels = len(self._channel_mults)

        for level in range(n_levels):
            for _ in range(self._num_res_blocks):
                h = self.down_blocks[block_idx](h, t_emb)   # ResBlock
                block_idx += 1
                # Apply AttentionBlock if present after this ResBlock
                if (block_idx < len(self.down_blocks)
                        and isinstance(self.down_blocks[block_idx], AttentionBlock)):
                    h = self.down_blocks[block_idx](h)
                    block_idx += 1
                skips.append(h)

            # Downsample between levels (not after the last level)
            if level < n_levels - 1:
                h = self.downsamplers[level](h)
                skips.append(h)

        # ── Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # ── Decoder: ResBlocks (+ optional Attention) interleaved with Upsamples
        block_idx = 0
        us_idx = 0

        for level, _ in reversed(list(enumerate(self._channel_mults))):
            for _ in range(self._num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[block_idx](h, t_emb)     # ResBlock
                block_idx += 1
                # Apply AttentionBlock if present after this ResBlock
                if (block_idx < len(self.up_blocks)
                        and isinstance(self.up_blocks[block_idx], AttentionBlock)):
                    h = self.up_blocks[block_idx](h)
                    block_idx += 1

            # Upsample back toward full resolution (not after the final output level)
            if level > 0:
                h = self.upsamplers[us_idx](h)
                us_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(image_channels=1, base_channels=64, image_size=256).to(device)

    x = torch.randn(2, 1, 256, 256).to(device)
    t = torch.rand(2).to(device)

    with torch.no_grad():
        out = model(x, t)

    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")   # should be [2, 1, 256, 256]

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params : {total_params:.1f}M")
