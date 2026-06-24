# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from numpy.lib.stride_tricks import sliding_window_view

from tqdm.auto import tqdm

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "IQL-D4RL"
    # wandb run name
    name: str = "FMIQL"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"
    # discount factor
    discount: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.7
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(1e4)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 2048
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    #  where to use dropout for policy network, optional
    actor_dropout: Optional[float] = None
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e2)
    # number of episodes to run during evaluation
    n_episodes: int = 3
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    #wandb
    use_wandb: bool = False

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    '''
    chunked
    '''
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        chunk_len: int = 4,
    ):
        self.chunk_len = chunk_len
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._states_chunk = torch.zeros(
            (buffer_size, chunk_len, state_dim), dtype=torch.float32, device=device
        )
        self._actions_chunk = torch.zeros(
            (buffer_size, chunk_len, action_dim), dtype=torch.float32, device=device
        )
        self._rewards_chunk = torch.zeros((buffer_size, chunk_len), dtype=torch.float32, device=device)
        self._next_states_chunk = torch.zeros(
            (buffer_size, chunk_len, state_dim), dtype=torch.float32, device=device
        )
        self._dones_chunk = torch.zeros((buffer_size, chunk_len), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray],gamma=0.99):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        obs = np.asarray(data["observations"])
        acts = np.asarray(data["actions"])
        rewards = np.asarray(data["rewards"], dtype=np.float32)
        next_obs = np.asarray(data["next_observations"])
        terminals = np.asarray(data["terminals"], dtype=np.bool_)

        T = obs.shape[0]
        k = self.chunk_len

        if T < k:
            raise ValueError("Dataset is shorter than chunk_len.")

        # [T-k+1, k, ...]
        obs_win = sliding_window_view(obs, window_shape=k, axis=0) # [N, obs_dim, k]
        obs_win = obs_win.transpose(0,2,1) # [N, k, obs_dim]
        obs_next_win = sliding_window_view(next_obs, window_shape=k, axis=0)
        obs_next_win = obs_next_win.transpose(0,2,1)
        act_win = sliding_window_view(acts, window_shape=k, axis=0)
        act_win = act_win.transpose(0,2,1)
        rew_win = sliding_window_view(rewards, window_shape=k, axis=0) # [N, k]
        done_win = sliding_window_view(terminals, window_shape=k, axis=0)

        # filter
        valid = ~done_win[:, :-1].any(axis=1)
        idx = np.flatnonzero(valid)
        n = idx.shape[0]

        if n > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        gamma_powers = (gamma ** np.arange(self.chunk_len, dtype=np.float32))[None, :] # [1,len]


        self._states_chunk[:n] = self._to_tensor(obs_win[idx]) #[N, chunk_len, obs_dim]
        self._actions_chunk[:n] = self._to_tensor(act_win[idx])  # [N, chunk_len, act_dim]
        self._rewards_chunk[:n] = self._to_tensor(rew_win[idx])
        self._next_states_chunk[:n] = self._to_tensor(obs_next_win[idx])
        self._dones_chunk[:n] = self._to_tensor(done_win[idx])
        self._size = n
        self._pointer = n

        print(f"Dataset size: {n}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states_chunk = self._states_chunk[indices]
        actions_chunk = self._actions_chunk[indices]
        rewards_chunk = self._rewards_chunk[indices]
        next_states_chunk = self._next_states_chunk[indices]
        dones_chunk = self._dones_chunk[indices]
        states = states_chunk.reshape(-1, self.state_dim)
        actions = actions_chunk.reshape(-1, self.action_dim)
        rewards = rewards_chunk.reshape(-1, 1)
        next_states = next_states_chunk.reshape(-1, self.state_dim)
        dones = dones_chunk.reshape(-1, 1)
        return [
        states, 
        actions, 
        rewards, 
        next_states, 
        dones,
        states_chunk,
        actions_chunk,
        rewards_chunk,
        next_states_chunk,
        dones_chunk
        ]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.net.eval()
    episode_rewards = []
    for i in range(n_episodes):
        env.seed(seed + i)
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action_chunk = actor.act(state, device)
            for h in range(action_chunk.shape[0]):
                action = action_chunk[h,:]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
        episode_rewards.append(episode_reward)

    actor.net.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#--- network

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

    def __init__(self, obs_dim: int, cond_dim: int,max_groups: int = 8) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(obs_dim, cond_dim, kernel_size=1)
        self.block1 = nn.Sequential(
            make_group_norm(cond_dim,max_groups),
            nn.SiLU(),
            nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            make_group_norm(cond_dim,max_groups),
            nn.SiLU(),
            nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
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
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

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
        self.up_fuse = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()


        # Symmetric with encoder: reverse all but the deepest level
        for i in range(len(self.down_dims) - 1, 0, -1):
            in_ch = self.down_dims[i] + self.down_dims[i - 1]
            out_ch = self.down_dims[i - 1]
            self.up_fuse.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            self.dec_blocks.append(
                FiLMResBlock1D(
                    in_channels=out_ch,
                    out_channels=out_ch,
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
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Bottleneck
        for block in self.mid_blocks:
            x = block(x, cond)

        # Decoder
        for i, (fuse, block) in enumerate(zip(self.up_fuse, self.dec_blocks)):
            skip = skips[-(i + 2)]  # skip corresponding to the matching encoder level
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            x = block(x, cond)

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

#---


#---



class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class FMPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, act_horizon=4, chunk_len=4, unet_dims=[128, 256, 512], cond_dim = 128, time_emb_dim = 128, dropout=0.1):
        super().__init__()
        self.act_horizon = min(4,act_horizon)
        self.net = FMTFnet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            horizon=self.act_horizon,
            cond_dim=cond_dim,
            time_emb_dim=time_emb_dim,
            dropout=0.1,
        ).to(device)
        self.device = device
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.chunk_len = chunk_len

    def sample_tau(self, batch_size: int, tau_min: float = 0.0, tau_max: float = 1.0) -> torch.Tensor:
        """Shifted Beta distribution biased toward lower tau (harder noise regime)."""
        device = self.device
        beta = torch.distributions.Beta(
            concentration1=torch.tensor(0.7, device=device),
            concentration0=torch.tensor(1.5, device=device),
        )
        tau = beta.sample((batch_size, 1))
        tau = tau_min + (tau_max - tau_min) * tau
        return tau.clamp(0.0, 1.0)

    def flow_match_loss(
        self,
        obs: torch.Tensor,
        target_act: torch.Tensor,
        noise_scale: float = 1.0,
        use_ema = False
    ):
        """Conditional flow matching with linear path x_tau = tau*x1 + (1-tau)*eps."""
        device = self.device
        model = self.shadow_model if use_ema else self.net
        obs_seq = obs.reshape(-1,1,self.obs_dim)
        B = obs_seq.size(0)
        tau = self.sample_tau(B)  # [B, 1]
        eps = torch.randn_like(target_act) * noise_scale

        tau_b = tau[:, :, None]  # [B, 1, 1]
        x_tau = tau_b * target_act + (1.0 - tau_b) * eps
        target_velocity = target_act - eps
        pred_velocity = model(obs_seq, x_tau, tau)

        err = (pred_velocity - target_velocity)**2
        loss = err.mean(dim=(1,2))   # [B]

        return loss

    @torch.no_grad()
    def sample_action_chunk(
        self,
        obs: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Iterative denoising / ODE-style sampling from noise to action chunk."""
        device = self.device
        model = self.net
        obs_seq = obs.reshape(-1,1,self.obs_dim)
        B = obs_seq.size(0)
        H = self.act_horizon
        act_dim = self.act_dim
        x = torch.randn(B, H, act_dim, device=device)
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = ts[i].expand(B, 1)
            dt = ts[i + 1] - ts[i]
            v = model(obs_seq, x, t)
            x = x + dt * v
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_chunk = self.sample_action_chunk(state)
        return action_chunk.squeeze(0).cpu().numpy()


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], -1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        chunk_len: int = 4,
        act_horizon: int = 4,
        ema_decay = 0.999,
        use_ema = True
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device
        self.chunk_len = chunk_len
        self.act_horizon = act_horizon

        self.ema = EMA(self.actor.net, decay=ema_decay) if use_ema else None

        self.gamma_powers = (discount ** torch.arange(self.chunk_len, dtype=torch.float32)).to(device)

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        '''
        observations [B, obs_dim]
        actions [B, act_dim]
        '''
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        observations_chunk: torch.Tensor,
        actions_chunk: torch.Tensor,
        log_dict: Dict,
    ):
        init_obs = observations_chunk[:, 0, :]

        with torch.no_grad():
            target_q = self.q_target(observations_chunk, actions_chunk)

        v = self.vf(observations_chunk)

        advs = target_q - v

        adv = (advs * self.gamma_powers).mean(dim=-1, keepdim=True) # [B, chunk_len] -> [B, 1]

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        fm_loss = self.actor.flow_match_loss(init_obs, actions_chunk)
        fm_loss = fm_loss.reshape(-1,1)

        policy_loss = torch.mean(exp_adv * fm_loss)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            obs_chunks,
            act_chunks,
            rew_chunks,
            next_obs_chunks,
            done_chunks
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(obs_chunks, act_chunks, log_dict)

        #self.ema.update(self.actor.net)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])

        self.total_it = state_dict["total_it"]


    def ema_copy(self):
        self.actor.shadow_model.load_state_dict(self.actor.net.state_dict())
        self.ema.copy_to(self.actor.shadow_model)
        self.actor.shadow_model.eval()


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset,gamma=config.discount)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = FMPolicy(
        state_dim, 
        action_dim, 
        config.device,
        act_horizon=4,
        chunk_len=4,
        unet_dims=[128, 256, 512], 
        cond_dim = 128, 
        time_emb_dim = 128,
        dropout=0.1
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.net.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training FMIQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if (config.use_wandb):
        wandb_init(asdict(config))

    evaluations = []

    max_score = -100

    pbar = tqdm(range(1, config.max_timesteps + 1), desc="training", dynamic_ncols=True)
    for t in pbar:
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        if (config.use_wandb):
            wandb.log(log_dict, step=trainer.total_it)

        pbar.set_postfix(
            {
                "epoch": f"{t}",
                "total_steps": f"{t*config.batch_size}",
                "v_loss": log_dict["value_loss"], 
                "q_loss": log_dict["q_loss"],
                "a_loss": log_dict["actor_loss"]
            }
        )
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            #trainer.ema_copy()
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            score_std = eval_scores.std()
            normalized_eval_score = env.get_normalized_score(eval_scores) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes:"
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score.mean():.3f} +- {normalized_eval_score.std()}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            if (config.use_wandb):
                wandb.log(
                    {
                        "d4rl_normalized_score": normalized_eval_score.mean(),
                        "episode return": eval_score,
                        "d4rl_normalized_score_std": normalized_eval_score.std()
                    }, step=trainer.total_it
                )

    if (config.use_wandb):
        wandb.finish()


if __name__ == "__main__":
    train()
