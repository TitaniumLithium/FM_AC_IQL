import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, List, Tuple
import math

def mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in state.items():
            if k not in self.shadow or not torch.is_floating_point(v):
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.shadow.items():
            if k in state and torch.is_floating_point(state[k]):
                state[k].copy_(v)

def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def sinusoidal_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """tau: [B,1] or [B]. returns [B, dim]."""
    if tau.dim() == 2 and tau.size(-1) == 1:
        tau = tau.squeeze(-1)
    half = dim // 2
    device = tau.device
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=tau.dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = tau[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ObsConditionEncoder(nn.Module):

    def __init__(self, obs_dim: int, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(obs_dim, d_model, kernel_size=1)
        self.block1 = nn.Sequential(
            make_group_norm(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            make_group_norm(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: [B, obs_T, obs_dim]
        x = obs_seq.transpose(1, 2)  # [B, obs_dim, obs_T]
        x = self.in_proj(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x.mean(dim=-1)  # [B, d_model]
        return self.out(x)


class FiLMResBlock1D(nn.Module):
    """Residual 1D block with FiLM conditioning from a context vector."""

    def __init__(self, channels: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = make_group_norm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = make_group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L], cond: [B,cond_dim]
        scale_shift = self.cond(cond).unsqueeze(-1)  # [B, 2C, 1]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.dropout(self.conv1(h))

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(self.conv2(h))
        return x + h


class FM1DUnet(nn.Module):
    """conditional flow-matching using a 1D U-Net backbone.

    Input:
      - normalized observation sequence [B, T, obs_dim]
      - noisy action chunk [B, horizon, act_dim]
      - scalar flow time tau [B, 1]

    Output:
      - predicted velocity field [B, horizon, act_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ctx_len: int,
        horizon: int,
        d_model: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ctx_len = ctx_len
        self.horizon = horizon
        self.d_model = d_model

        self.obs_encoder = ObsConditionEncoder(obs_dim=obs_dim, d_model=d_model)
        self.tau_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.act_in = nn.Sequential(
            nn.Conv1d(act_dim, d_model, kernel_size=1),
            nn.GroupNorm(num_groups=max(1, min(8, d_model)), num_channels=d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

        cond_dim = d_model

        # U-Net encoder path
        self.enc1 = FiLMResBlock1D(d_model, cond_dim, dropout=dropout)
        self.down1 = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMResBlock1D(d_model, cond_dim, dropout=dropout)
        self.down2 = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = FiLMResBlock1D(d_model, cond_dim, dropout=dropout)
        self.mid2 = FiLMResBlock1D(d_model, cond_dim, dropout=dropout)

        # U-Net decoder path
        self.up2_fuse = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.dec2 = FiLMResBlock1D(d_model, cond_dim, dropout=dropout)
        self.up1_fuse = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.dec1 = FiLMResBlock1D(d_model, cond_dim, dropout=dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, min(8, d_model)), num_channels=d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, act_dim, kernel_size=1),
        )

    def forward(self, obs_seq: torch.Tensor, noisy_act: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # obs_seq: [B, T, obs_dim]
        # noisy_act: [B, horizon, act_dim]
        # tau: [B, 1]
        cond = self.obs_encoder(obs_seq)  # [B, d_model]
        tau_emb = sinusoidal_embedding(tau, self.d_model)
        cond = cond + self.tau_mlp(tau_emb)

        x = noisy_act.transpose(1, 2)  # [B, act_dim, horizon]
        x = self.act_in(x)             # [B, d_model, horizon]

        skip1 = self.enc1(x, cond)      # [B, d_model, horizon]
        x = self.down1(skip1)           # [B, d_model, ceil(horizon/2)]

        skip2 = self.enc2(x, cond)      # [B, d_model, ...]
        x = self.down2(skip2)           # [B, d_model, ...]

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        x = F.interpolate(x, size=skip2.size(-1), mode="linear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2_fuse(x)
        x = self.dec2(x, cond)

        x = F.interpolate(x, size=skip1.size(-1), mode="linear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1_fuse(x)
        x = self.dec1(x, cond)

        velocity = self.out(x).transpose(1, 2)  # [B, horizon, act_dim]
        return velocity

