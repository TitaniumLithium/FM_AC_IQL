import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, List, Tuple, Sequence, Optional
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
    def __init__(self, obs_dim: int, cond_dim: int, max_groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: [B, obs_h, obs_dim]
        x = obs_seq.mean(dim=1)  # [B, obs_dim], works for obs_h=1 and >1
        return self.net(x)


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad/crop temporal length to target_len."""
    cur_len = x.size(-1)
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - cur_len))



class FiLMResBlock1D(nn.Module):
    """
    Residual 1D block with FiLM conditioning from a context vector.
    x:    [B, C, L]
    cond: [B, cond_dim]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding."
        padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = make_group_norm(in_channels, max_groups=max_groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = make_group_norm(out_channels, max_groups=max_groups)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * out_channels),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L], cond: [B,cond_dim]
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.cond(cond).unsqueeze(-1)  # [B, 2*out_channels, 1]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


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
        horizon: int,
        dropout: float = 0.1,
        down_dims: Sequence[int] = (128, 256, 512),
        kernel_size: int = 3,
        cond_dim: int = 128,
        time_emb_dim: int = 128,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(down_dims) < 1:
            raise ValueError("down_dims must contain at least one channel size.")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.down_dims = list(down_dims)
        self.kernel_size = kernel_size

        base_dim = down_dims[0]
        cond_dim = cond_dim or base_dim
        time_emb_dim = time_emb_dim or base_dim

        self.obs_encoder = ObsConditionEncoder(obs_dim=obs_dim, cond_dim=cond_dim)
        self.tau_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_fuse = nn.Linear(cond_dim, cond_dim)

        in_padding = kernel_size // 2
        self.act_in = nn.Sequential(
            nn.Conv1d(act_dim, base_dim, kernel_size=kernel_size, padding=in_padding),
            make_group_norm(base_dim, max_groups=max_groups),
            nn.SiLU(),
            nn.Conv1d(base_dim, base_dim, kernel_size=kernel_size, padding=in_padding),
        )

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()


        curr_channels = base_dim
        for i, ch in enumerate(self.down_dims):
            self.enc_blocks.append(
                FiLMResBlock1D(
                    in_channels=curr_channels,
                    out_channels=ch,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    max_groups=max_groups,
                )
            )
            curr_channels = ch

            if i < len(self.down_dims) - 1:
                next_ch = self.down_dims[i + 1]
                # Standard strided conv downsample
                self.downsamples.append(
                    nn.Conv1d(curr_channels, next_ch, kernel_size=4, stride=2, padding=1)
                )
                curr_channels = next_ch


        # Bottleneck
        self.mid_blocks = nn.ModuleList(
            [
                FiLMResBlock1D(
                    in_channels=self.down_dims[-1],
                    out_channels=self.down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    max_groups=max_groups,
                ),
                FiLMResBlock1D(
                    in_channels=self.down_dims[-1],
                    out_channels=self.down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    max_groups=max_groups,
                ),
            ]
        )


        # Decoder
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()


        # Symmetric with encoder: reverse all but the deepest level
        for i in range(len(self.down_dims) - 1, 0, -1):
            deep_ch = self.down_dims[i]
            skip_ch = self.down_dims[i - 1]

            self.upsamples.append(
                nn.ConvTranspose1d(deep_ch, skip_ch, kernel_size=4, stride=2, padding=1)
            )
            self.dec_blocks.append(
                FiLMResBlock1D(
                    in_channels=skip_ch * 2,
                    out_channels=skip_ch,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    max_groups=max_groups,
                )
            )

        self.out = nn.Sequential(
            make_group_norm(self.down_dims[0], max_groups=max_groups),
            nn.SiLU(),
            nn.Conv1d(self.down_dims[0], act_dim, kernel_size=1),
        )

    def _build_condition(self, obs_seq: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: [B, T, obs_dim]
        tau:     [B, 1] or [B]
        """
        cond = self.obs_encoder(obs_seq)  # [B, cond_dim]

        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        tau_emb = sinusoidal_embedding(tau, self.tau_mlp[0].in_features)
        tau_cond = self.tau_mlp(tau_emb)

        cond = self.cond_fuse(cond + tau_cond)
        return cond

    def forward(self, obs_seq: torch.Tensor, noisy_act: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # obs_seq: [B, T, obs_dim]
        # noisy_act: [B, horizon, act_dim]
        # tau: [B, 1]
        cond = self._build_condition(obs_seq, tau)

        x = noisy_act.transpose(1, 2)  # [B, horizon, act_dim] -> [B, act_dim, horizon]
        x = self.act_in(x)             # [B, act_dim, horizon]

        skips: List[torch.Tensor] = []

        # Encoder
        for i, block in enumerate(self.enc_blocks):
            x = block(x, cond)
            if i < len(self.enc_blocks) - 1:
                skips.append(x)
                x = self.downsamples[i](x)

        # Bottleneck
        for block in self.mid_blocks:
            x = block(x, cond)

        # Decoder
        for skip, upsample, block in zip(reversed(skips), self.upsamples, self.dec_blocks):
            x = upsample(x)                         # [B, skip_ch, ?]
            x = _match_length(x, skip.size(-1))     # temporal alignment
            x = torch.cat([x, skip], dim=1)         # [B, 2*skip_ch, L]
            x = block(x, cond)                      # [B, skip_ch, L]

        velocity = self.out(x).transpose(1, 2)  # [B, horizon, act_dim]
        return velocity


class FMTFnet(nn.Module):
    """conditional flow-matching using a TF backbone.

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
        horizon: int,
        dropout: float = 0.1,
        cond_dim: int = 128,
        time_emb_dim: int = 128,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.time_emb_dim = time_emb_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.act_encoder = nn.Sequential(
            nn.Linear(act_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.tau_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.obs_pos = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        self.act_pos = nn.Parameter(torch.randn(1, horizon, cond_dim) * 0.02)
        self.ctx_summary = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cond_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.final = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor, noisy_act: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # obs: [B, 1, obs_dim]
        # noisy_act: [B, horizon, act_dim]
        # tau: [B, 1]
        B = obs.shape[0]

        obs_tokens = self.obs_encoder(obs) + self.obs_pos[:, : obs.size(1)]
        obs_tokens = self.blocks(obs_tokens)
        ctx = obs_tokens[:, -1:, :]  # summarize with the most recent observation token

        tau_emb = sinusoidal_embedding(tau, self.time_emb_dim)
        tau_emb = self.tau_mlp(tau_emb).unsqueeze(1)  # [B,1,C]

        act_tokens = self.act_encoder(noisy_act) + self.act_pos[:, : noisy_act.size(1)]
        act_tokens = act_tokens + tau_emb

        # Prefix-style conditioning, similar in spirit to a context block.
        prefix = ctx + self.ctx_summary
        x = torch.cat([prefix, act_tokens], dim=1)
        x = self.blocks(x)
        act_hidden = x[:, 1:, :]
        velocity = self.final(act_hidden)
        return velocity