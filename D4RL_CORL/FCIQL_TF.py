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

import imageio

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
    beta: float = 2.5
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.7
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(1e5)
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
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
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
    horizon: int = 4
    chunk_len: int = 4
    actor_train_start_ratio: float = 0.0
    cfg_tune: float = 1.0
    cfg_weight: float = 1.5
    cfg_drop: float = 0.1
    fm_steps: int = 10

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

def unnormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return data * std + mean


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

def wrap_env_gymnasium(
    env,
    gym,
    observation_space,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
):
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state,observation_space=observation_space)
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
        self.action_horizon = chunk_len
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

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
        self._terminal_chunk = torch.zeros((buffer_size, chunk_len), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def dataset_to_episodes(self, data: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        obs = np.asarray(data["observations"], dtype=np.float32)
        acts = np.asarray(data["actions"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32)
        next_obs = np.asarray(data["next_observations"], dtype=np.float32)
        terminals = np.asarray(data["terminals"], dtype=np.bool_)
        
        trunactions = np.asarray(data["timeouts"], dtype=np.bool_)
        
        dones = np.logical_or(terminals,trunactions)

        episodes = []
        episode = {"observations": [], "actions": [], "rewards": [], "next_observations": [], "terminals": [], "trunactions": []}
        for i in range(len(obs)):
            episode["observations"].append(obs[i])
            episode["actions"].append(acts[i])
            episode["rewards"].append(rewards[i])
            episode["next_observations"].append(next_obs[i])
            episode["terminals"].append(terminals[i])
            episode["trunactions"].append(trunactions[i])
            if dones[i]:
                episodes.append({k: np.array(v) for k, v in episode.items()})
                episode = {"observations": [], "actions": [], "rewards": [], "next_observations": [], "terminals": [], "trunactions": []}
        return episodes

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray],gamma=0.99):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        episodes = self.dataset_to_episodes(data)

        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        next_obs_list: List[np.ndarray] = []
        rew_list: List[np.ndarray] = []
        done_list: List[np.ndarray] = []
        all_act_list = np.asarray(data["actions"], dtype=np.float32)
        all_obs_list = np.asarray(data["observations"], dtype=np.float32)

        #gamma_powers = (gamma ** np.arange(self.action_horizon, dtype=np.float32)).astype(np.float32)

        for ep in episodes:
            obs = ep["observations"]# [T, obs_dim]
            next_obs = ep["next_observations"] # [T, obs_dim]
            actions = ep["actions"] # [T, act_dim]
            rewards = ep["rewards"] # [T]
            terminations = ep["terminals"]
            truncations = ep["trunactions"]
            dones = np.logical_or(terminations, truncations)

            T = actions.shape[0]
            if T < self.action_horizon:
                continue

            # Sliding window inside the episode.
            # start t: 0 .. T-horizon

            for t in range(T - self.action_horizon + 1):
                end = t + self.action_horizon

                # Optional safety check:
                # do not let the chunk cross an earlier terminal/truncation.
                # In standard episode data this usually will not happen except at the end.
                if np.any(dones[t:end - 1]):
                    continue

                #chunk_reward = float(np.sum(rewards[t:end] * gamma_powers))
                #chunk_done = float(dones[end - 1])

                obs_list.append(obs[t:end])                     # starting state
                act_list.append(actions[t:end])             # [horizon, act_dim]
                next_obs_list.append(next_obs[t:end])              # state after chunk
                rew_list.append(rewards[t:end])             # [horizon]
                # test for dones
                done_list.append(terminations[t:end])

        obs_arr = np.stack(obs_list, axis=0)                    # [N, horizon, obs_dim]
        act_arr = np.stack(act_list, axis=0)                   # [N, horizon, act_dim]
        next_obs_arr = np.stack(next_obs_list, axis=0)         # [N, horizon, obs_dim]
        rew_arr = np.stack(rew_list, axis=0)   # [N, horizon]
        done_arr = np.stack(done_list, axis=0)  # [N, horizon]

        T = obs_arr.shape[0]
        k = self.action_horizon

        if T < k:
            raise ValueError("Dataset is shorter than chunk_len.")

        n = min(T, self._buffer_size)

        self._states_chunk[:n] = self._to_tensor(obs_arr) #[N, chunk_len, obs_dim]
        self._actions_chunk[:n] = self._to_tensor(act_arr)  # [N, chunk_len, act_dim]
        self._rewards_chunk[:n] = self._to_tensor(rew_arr)
        self._next_states_chunk[:n] = self._to_tensor(next_obs_arr)
        self._terminal_chunk[:n] = self._to_tensor(done_arr)
        self._size = n
        self._pointer = n

        print(f"Dataset size: {n}")

    def load_minari_dataset(self, dataset):
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        next_obs_list: List[np.ndarray] = []
        rew_list: List[np.ndarray] = []
        done_list: List[np.ndarray] = [] 
        terminal_list: List[np.ndarray] = []
        all_act_list= []
        all_obs_list = []

        horizon = self.action_horizon

        for ep in dataset.iterate_episodes():
            obs = np.asarray(ep.observations, dtype=np.float32)
            actions = np.asarray(ep.actions, dtype=np.float32)
            rewards = np.asarray(ep.rewards, dtype=np.float32)
            terminations = np.asarray(ep.terminations, dtype=np.bool_)
            truncations = np.asarray(ep.truncations, dtype=np.bool_)
            dones = np.logical_or(terminations, truncations)

            #flatten rewards,ter,done to 1D
            rewards = rewards.flatten()
            terminations = terminations.flatten()
            dones = dones.flatten()

            all_act_list.append(actions)
            all_obs_list.append(obs)

            T = actions.shape[0]
            if T < horizon:
                continue

            for t in range(T - horizon + 1):
                end = t + horizon

                # Optional safety check:
                # do not let the chunk cross an earlier terminal/truncation.
                # In standard episode data this usually will not happen except at the end.
                if np.any(dones[t:end - 1]):
                    continue

                obs_list.append(obs[t:end])                     # starting state
                act_list.append(actions[t:end])             # [horizon, act_dim]
                next_obs_list.append(obs[t+1:end+1])              # state after chunk
                rew_list.append(rewards[t:end])             # [horizon]
                done_list.append(dones[t:end])
                terminal_list.append(terminations[t:end])

        if len(obs_list) == 0:
            raise ValueError(
                f"No valid chunk samples found."
            )

        obs_arr = np.stack(obs_list, axis=0)
        act_arr = np.stack(act_list, axis=0)
        next_obs_arr = np.stack(next_obs_list, axis=0)
        rew_arr = np.stack(rew_list, axis=0)
        done_arr = np.stack(done_list, axis=0)
        terminal_arr = np.stack(terminal_list, axis=0)

        all_act_arr = np.concatenate(all_act_list, axis=0) # [N, act_dim]
        all_obs_arr = np.concatenate(all_obs_list, axis=0) # [N, obs_dim]

        assert obs_arr.shape[0] ==  act_arr.shape[0] == next_obs_arr.shape[0] == rew_arr.shape[0] == terminal_arr.shape[0]

        print(obs_arr.shape,act_arr.shape,done_arr.shape)

        obs_mean,obs_std = compute_mean_std(all_obs_arr,eps=1e-5)
        act_mean,act_std = compute_mean_std(all_act_arr,eps=1e-5)

        n_obs = (obs_arr - obs_mean) / obs_std
        n_obs_next = (next_obs_arr - obs_mean) / obs_std
        n_act = (act_arr - act_mean) / act_std

        n = min(obs_arr.shape[0], self._buffer_size)

        self._states_chunk[:n] = self._to_tensor(n_obs) #[N, chunk_len, obs_dim]
        self._actions_chunk[:n] = self._to_tensor(n_act)  # [N, chunk_len, act_dim]
        self._rewards_chunk[:n] = self._to_tensor(rew_arr)
        self._next_states_chunk[:n] = self._to_tensor(n_obs_next)
        self._terminal_chunk[:n] = self._to_tensor(terminal_arr)
        self._size = n
        self._pointer = n

        print(f"Dataset size: {n}")

        return obs_mean,obs_std,act_mean,act_std

        
    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states_chunk = self._states_chunk[indices]
        actions_chunk = self._actions_chunk[indices]
        rewards_chunk = self._rewards_chunk[indices]
        next_states_chunk = self._next_states_chunk[indices]
        terminal_chunk = self._terminal_chunk[indices]
        return [states_chunk, actions_chunk, rewards_chunk, next_states_chunk, terminal_chunk]

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
    #wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, chunk_len=4,save_video=False,max_timesteps=400,new_gym=False,normalize_info = None,num_steps = 10
) -> np.ndarray:
    #env.seed(seed)
    actor.net.eval()
    episode_rewards = []
    infos = []
    writer = imageio.get_writer(
    "./video.mp4",
    fps=int(env.metadata.get("render_fps", 30)),
    codec="libx264"
    )
    for i in range(n_episodes):
        if new_gym:
            state, _ = env.reset(seed=seed+i)
            done = False
        else:
            env.seed(seed + i)
            state, done = env.reset(), False

        episode_reward = 0.0
        step = 0
        max_coverage = 0.0
        while not done:
            action_chunk = actor.act(state, device,num_steps=num_steps)
            if normalize_info:
                mean, std = normalize_info["act_mean"], normalize_info["act_std"]
                action_chunk = unnormalize(action_chunk,mean,std)
            chunk_len = min(chunk_len,action_chunk.shape[0])
            for h in range(chunk_len):
                action = action_chunk[h,:]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                if new_gym:
                    state, reward, terminal, trunc, info = env.step(action)
                    done = terminal or trunc
                    converage = info.get("coverage",0)
                    max_coverage = max(max_coverage,converage)
                    if i==n_episodes - 1 and save_video:
                        frame = env.render()
                        writer.append_data(frame)
                else:
                    state, reward, done, info = env.step(action)
                episode_reward += reward
                step+=1
                if done or step>=max_timesteps:# or np.random.uniform()>(h+1)/chunk_len:
                    break
            if step>=max_timesteps:
                break

        info["max_coverage"] = max_coverage
        episode_rewards.append(episode_reward)
        infos.append(info)
    
    writer.close()
    actor.net.train()
    return np.asarray(episode_rewards),infos


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
    """
    Lightweight condition encoder.
    """
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
    Residual 1D block with FiLM conditioning.
    Conditioning is injected AFTER the first conv, BEFORE the second conv.
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
        # x: [B, C, L], cond: [B, cond_dim]
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
    

class ConvNet1D(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        dropout: float = 0.0,
        cond_dim: int = 64,
        max_groups: int = 8,
        kernel_size: int = 4,
        down_dims: Sequence[int] = (64, 128),
        ):
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

        self.encoders = nn.Sequential()

        curr_channels = self.act_dim

        for i, ch in enumerate(self.down_dims):
            next_ch = ch
            self.encoders.append(
                nn.Conv1d(curr_channels, next_ch, kernel_size=self.kernel_size, stride=2, padding=1),
            )
            self.encoders.append(
                nn.SiLU(),
            )
            curr_channels = next_ch

        self.encoders.append(nn.AdaptiveAvgPool1d(1))

        self.final_mlp = nn.Sequential(
            nn.Linear(curr_channels+self.obs_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, 1),
        )

    def forward(self,obs,act_chunk):
        x = act_chunk.transpose(1, 2)  # [B, act_dim, horizon]
        x = self.encoders(x)
        x = x.squeeze(-1)
        mid = torch.cat([x,obs],dim=-1)
        out = self.final_mlp(mid)
        return out.flatten()




class FM1DUnet(nn.Module):
    """
    Lightweight conditional flow-matching 1D U-Net.

    Input:
      - obs_seq:   [B, obs_h, obs_dim]
      - noisy_act:  [B, horizon, act_dim]
      - tau:       [B, 1] or [B]

    Output:
      - velocity:   [B, horizon, act_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        dropout: float = 0.0,
        down_dims: Sequence[int] = (64, 128),
        kernel_size: int = 3,
        cond_dim: int = 64,
        time_emb_dim: int = 64,
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

        self.score_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.null_emb = nn.Parameter(torch.zeros((cond_dim,), dtype=torch.float32), requires_grad=True)
        nn.init.normal_(self.null_emb,std=0.01)
        
        self.cond_fuse = nn.Sequential(
            nn.Linear(3 * cond_dim, 2 * cond_dim),
            nn.SiLU(),
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        in_padding = kernel_size // 2
        self.act_in = nn.Sequential(
            nn.Conv1d(act_dim, base_dim, kernel_size=kernel_size, padding=in_padding),
            make_group_norm(base_dim, max_groups=max_groups),
            nn.SiLU(),
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
                self.downsamples.append(
                    nn.Conv1d(curr_channels, next_ch, kernel_size=4, stride=2, padding=1)
                )
                curr_channels = next_ch

        # Bottleneck: one block
        self.mid_blocks = nn.ModuleList(
            [
                FiLMResBlock1D(
                    in_channels=self.down_dims[-1],
                    out_channels=self.down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    max_groups=max_groups,
                )
            ]
        )

        # Decoder: transpose conv upsample + concat skip
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

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

    def _build_condition(self, obs_seq: torch.Tensor, tau: torch.Tensor, tune: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        obs_cond = self.obs_encoder(obs_seq)  # [B, cond_dim]

        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        tau_emb = sinusoidal_embedding(tau, self.tau_mlp[0].in_features)
        tau_cond = self.tau_mlp(tau_emb)

        score_cond = self.score_mlp(tune) if tune is not None else torch.zeros_like(obs_cond)

        if mask is not None:
            score_cond = torch.where(mask, score_cond, self.null_emb)
            # mask = True for valid tune, False for invalid tune

        cond = self.cond_fuse(torch.cat([obs_cond,tau_cond,score_cond],dim=-1))
        return cond

    def forward(self, obs_seq: torch.Tensor, noisy_act: torch.Tensor, tau: torch.Tensor, tune: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        cond = self._build_condition(obs_seq, tau, tune, mask)

        x = noisy_act.transpose(1, 2)  # [B, act_dim, horizon]
        x = self.act_in(x)

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

        velocity = self.out(x).transpose(1, 2)      # [B, horizon, act_dim]
        return velocity

#---


#---

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
        self.score_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.null_emb = nn.Parameter(torch.zeros((1,cond_dim), dtype=torch.float32), requires_grad=True)
        nn.init.normal_(self.null_emb,std=0.01)

        self.obs_pos = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        self.act_pos = nn.Parameter(torch.randn(1, horizon, cond_dim) * 0.02)
        self.score_pos = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        #self.ctx_summary = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02) not use

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

    def forward(self, obs: torch.Tensor, noisy_act: torch.Tensor, tau: torch.Tensor, tune: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # obs: [B, obs_dim]
        # noisy_act: [B, horizon, act_dim]
        # tau: [B, 1]
        # tune [B, 1]
        # mask [B, 1]
        B = obs.shape[0]

        obs_tokens = self.obs_encoder(obs) # [B,1,C]
        obs_tokens = obs_tokens + self.obs_pos

        tau_emb = sinusoidal_embedding(tau, self.time_emb_dim)
        tau_emb = self.tau_mlp(tau_emb).unsqueeze(1)  # [B,1,C]

        act_tokens = self.act_encoder(noisy_act) + self.act_pos[:, :noisy_act.size(1)] + tau_emb

        score_tokens = self.score_mlp(tune).unsqueeze(1) if tune is not None else torch.zeros_like(tau_emb)
        # [B,1,C]

        if mask is not None:
            mask = mask.unsqueeze(-1)
            score_tokens = torch.where(mask, score_tokens, self.null_emb)
            # mask = True for valid tune, False for invalid tune

        score_tokens = score_tokens + self.score_pos

        
        x = torch.cat([obs_tokens, score_tokens, act_tokens], dim=1)
        x = self.blocks(x)
        action_tokens = x[:, 2:]
        velocity = self.final(action_tokens)
        return velocity

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
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, act_horizon=4, chunk_len=4, unet_dims=[128, 256, 512], cond_dim = 128, time_emb_dim = 128, dropout=0,cfg_tune=1.0,cfg_weight=1.5):
        super().__init__()
        self.act_horizon = act_horizon
        self.net = FMTFnet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            horizon=self.act_horizon,
            cond_dim=cond_dim,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        ).to(device)
        self.shadow_model = FMTFnet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            horizon=self.act_horizon,
            cond_dim=cond_dim,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        ).to(device)
        self.device = device
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.chunk_len = chunk_len
        self.cfg_tune = cfg_tune
        self.cfg_weight = cfg_weight

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

    def flow_match_loss_cfg(
        self,
        obs: torch.Tensor,
        target_act: torch.Tensor,
        tune: torch.Tensor,
        mask: torch.Tensor,
        noise_scale: float = 1.0,
        use_ema = False
    ):
        """Conditional flow matching with linear path x_tau = tau*x1 + (1-tau)*eps."""
        device = self.device
        model = self.net
        obs_seq = obs.reshape(-1,1,self.obs_dim)
        B = obs_seq.size(0)
        tau = self.sample_tau(B)  # [B, 1]
        eps = torch.randn_like(target_act) * noise_scale

        tau_b = tau[:, :, None]  # [B, 1, 1]
        x_tau = tau_b * target_act + (1.0 - tau_b) * eps
        target_velocity = target_act - eps

        # test Q_grad
        # target_velocity
        #Q_grad = Q(s,x_tau)

        pred_velocity = model(obs_seq, x_tau, tau,tune=tune,mask=mask)

        err = (pred_velocity - target_velocity)**2
        loss = err.mean(dim=(1,2))   # [B]

        return loss

    @torch.no_grad()
    def sample_action_chunk_cfg(
        self,
        obs: torch.Tensor,
        num_steps: int = 10,
        net=None,
        cfg_tune = 1.0,
        cfg_weight = 1.5,
    ) -> torch.Tensor:
        """Iterative denoising / ODE-style sampling from noise to action chunk."""
        device = self.device
        if net is None:
            model = self.net
        else:
            model = net
        obs_seq = obs.reshape(-1,1,self.obs_dim)
        B = obs_seq.size(0)
        H = self.act_horizon
        act_dim = self.act_dim
        x = torch.randn(B, H, act_dim, device=device)
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        tune = torch.ones((B, 1), device=device) * cfg_tune
        mask = torch.ones((B, 1),device=device,dtype=torch.bool)
        for i in range(num_steps):
            t = ts[i].expand(B, 1)
            dt = ts[i + 1] - ts[i]
            v_c = model(obs_seq, x, t,tune=tune,mask=mask)
            v_unc = model(obs_seq, x, t,tune=tune,mask=torch.zeros_like(mask))
            v = v_unc + cfg_weight*(v_c-v_unc)
            x = x + dt * v
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu",ema = True, tune = None, cfg_weight=None,num_steps=None):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        net = self.shadow_model if ema else self.net
        if tune is None:
            tune = self.cfg_tune
        if cfg_weight is None:
            cfg_weight = self.cfg_weight
        action_chunk = self.sample_action_chunk_cfg(state,net=net,cfg_tune=tune, cfg_weight=cfg_weight, num_steps=num_steps)
        return action_chunk.squeeze(0).cpu().numpy()


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2,chunk_len: int = 4
    ):
        super().__init__()
        dims = [state_dim + action_dim * chunk_len, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.chunk_dim = action_dim * chunk_len

    def both(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        action_chunk = action_chunk.reshape(B,-1)
        sa = torch.cat([state, action_chunk], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        '''
        obs [B, obs_dim]
        action_chunk [B, H, act_dim]
        '''
        B = state.shape[0]
        action_chunk = action_chunk.reshape(B,-1)
        return torch.min(*self.both(state, action_chunk))


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
        use_ema = True,
        cfg_drop = 0.1
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
        self.grad_clip_norm = 1.0

        self.cfg_drop = cfg_drop

        self.ema = EMA(self.actor.net, decay=ema_decay) if use_ema else None

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        '''
        obs [B, obs_dim]
        actions [B, H, act_dim]
        '''
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        log_dict["v_mean"] = v.mean().item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip_norm)
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
        chunk_gamma = self.discount ** self.act_horizon
        chunk_discounts = self.discount ** torch.arange(self.act_horizon,device=self.device)
        chunk_rewards = (rewards * chunk_discounts[None,:]).sum(dim=1) # [B]
        chunk_rewards = chunk_rewards/self.act_horizon
        chunk_terminals = terminals.any(dim=1)
        targets = chunk_rewards + (1.0 - chunk_terminals.float()) * chunk_gamma * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        log_dict["q_mean"] = qs[0].mean().item()
        log_dict["q_std"] = qs[0].std().item()
        log_dict["targets_std"] = targets.std().item()
        log_dict["targets_mean"] = targets.mean().item()
        log_dict["chunk_rewards"] = chunk_rewards.mean().item()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.qf.parameters(), self.grad_clip_norm)
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        
        #adv = (adv - adv.mean()) / (adv.std()+1e-7)

        #exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        #adv = adv.clamp(-5.0, 5.0)

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        tune = torch.tanh(exp_adv.detach() / self.beta).reshape(-1,1)

        p_drop = self.cfg_drop
        B = tune.shape[0]

        mask = (torch.rand(B,1,device=self.actor.device) > p_drop) # [B,1]
        # p_drop = 0.1, so 10% of the weights are masked out (set to zero) during training.

        fm_loss = self.actor.flow_match_loss_cfg(observations, actions, tune=tune, mask=mask)
        fm_loss = fm_loss.reshape(-1,1)

        policy_loss = torch.mean(fm_loss)
        log_dict["actor_loss"] = policy_loss.item()

        log_dict["weight_mean"] = tune.mean().item()
        log_dict["weight_std"] = tune.std().item()
        log_dict["weight_max"] = tune.max().item()
        log_dict["weight_min"] = tune.min().item()

        with torch.no_grad():
            ess = (
                exp_adv.sum() ** 2 /
                (exp_adv.pow(2).sum()+1e-8)
            )

            ess_ratio = ess / exp_adv.shape[0]

        log_dict["ess_ratio"] = ess_ratio.item()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch, train_actor = True) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        init_obs = observations[:,0,:]
        final_obs = next_observations[:,-1,:]

        with torch.no_grad():
            next_v = self.vf(final_obs)
        # Update value function
        adv = self._update_v(init_obs, actions, log_dict)
        #rewards = rewards.squeeze(dim=-1)
        #dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, init_obs, actions, rewards, dones, log_dict)
        # Update actor

        if train_actor:

            self._update_policy(adv, init_obs, actions, log_dict)

            self.ema.update(self.actor.net)

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
    if "pusht" in config.env:
        import sys
        sys.path.append("./")
        from algorithms.offline.minari_loader import load_minari_dataset,load_env_minari
        import gymnasium as gym
        env,dataset = load_env_minari(config.env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        import gym
        env = gym.make(config.env)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        dataset = env.get_dataset()
        
        if "next_observations" not in dataset:
            next_obs = dataset["observations"][1:]
            dataset["next_observations"] = next_obs
            dataset["observations"] = dataset["observations"][:-1]
            dataset["actions"] = dataset["actions"][:-1]
            dataset["timeouts"] = dataset["timeouts"][:-1]
            dataset["terminals"] = dataset["terminals"][:-1]
            dataset["rewards"] = dataset["rewards"][:-1]

            assert len(dataset["next_observations"]) == len(dataset["observations"]) == len(dataset["actions"]) == len(dataset["timeouts"]) == len(dataset["terminals"]) == len(dataset["rewards"])
    

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

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
        chunk_len=config.horizon
    )

    act_normalize_info = {}

    if "pusht" in config.env:
        state_mean,state_std,act_mean,act_std = replay_buffer.load_minari_dataset(dataset)
        obs_space = env.observation_space
        env = wrap_env_gymnasium(env, gym, obs_space, state_mean=state_mean, state_std=state_std)
        act_normalize_info["act_mean"] = act_mean
        act_normalize_info["act_std"] = act_std

    else:
        env = wrap_env(env, state_mean=state_mean, state_std=state_std)

        act_mean, act_std = compute_mean_std(dataset["actions"], eps=1e-5)
        dataset["actions"] = normalize_states(
            dataset["actions"], act_mean, act_std
        )
        act_normalize_info["act_mean"] = act_mean
        act_normalize_info["act_std"] = act_std

        replay_buffer.load_d4rl_dataset(dataset,gamma=config.discount)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    if "pusht" in config.env:
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim,chunk_len=config.horizon).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = FMPolicy(
        state_dim, 
        action_dim, 
        config.device,
        act_horizon=config.horizon,
        chunk_len=config.chunk_len,
        unet_dims=[128, 256], 
        cond_dim = 128,
        time_emb_dim = 128,
        dropout=0.0,
        cfg_tune=config.cfg_tune,
        cfg_weight=config.cfg_weight
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
        "act_horizon": config.horizon,
        "chunk_len": config.chunk_len,
        "cfg_drop": config.cfg_drop
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
        train_actor = t > config.max_timesteps * config.actor_train_start_ratio
        log_dict = trainer.train(batch,train_actor)
        if (config.use_wandb):
            wandb.log(log_dict, step=trainer.total_it)

        pbar.set_postfix(
            {
                "epoch": f"{t}",
                "total_steps": f"{t*config.batch_size}",
                "v_loss": log_dict["value_loss"], 
                "q_loss": log_dict["q_loss"],
                "a_loss": log_dict.get("actor_loss",0),
                "q_mean": log_dict["q_mean"],
                "v_mean": log_dict["v_mean"],
                "target_q": log_dict["targets_mean"],
                "chunk_rewards": log_dict.get("chunk_rewards",0),
                "ess_ratio" : log_dict.get("ess_ratio",0)
            }
        )
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0 and train_actor:
            print(f"Time steps: {t + 1}")
            trainer.ema_copy()
            eval_scores, infos = eval_actor(
                env,
                actor,
                chunk_len=config.chunk_len,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                max_timesteps=400 if "pusht" in config.env else 1000,
                new_gym=True if "pusht" in config.env else False,
                save_video=True if "pusht" in config.env else False,
                normalize_info=act_normalize_info if len(act_normalize_info)>0 else None,
                num_steps = config.fm_steps
            )
            eval_score = eval_scores.mean()
            score_std = eval_scores.std()
            if "pusht" not in config.env:
                normalized_eval_score = env.get_normalized_score(eval_scores) * 100.0
            else:
                normalized_eval_score = torch.zeros((config.n_episodes,))
                for i,info in enumerate(infos):
                    print(info["success"],info.get("max_coverage",0))
                    if info["success"]:
                        normalized_eval_score[i] += 1
                if config.use_wandb:
                    wandb.log({
                        "video": wandb.Video(
                            "./video.mp4",
                            fps=30,
                            format="mp4"
                        )
                    }, step=trainer.total_it)
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes:"
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score.mean():.3f} +- {normalized_eval_score.std():.3f}"
                F"  max={normalized_eval_score.max():.3f} min={normalized_eval_score.min():.3f}"
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
                        "d4rl_normalized_score_max": normalized_eval_score.max(),
                        "d4rl_normalized_score_min": normalized_eval_score.min(),
                        "episode return": eval_score,
                        "d4rl_normalized_score_std": normalized_eval_score.std(),
                        "ess_ratio": log_dict["ess_ratio"]
                    }, step=trainer.total_it
                )
            for key,log in log_dict.items():
                print(f"{key} : {log}")

    if (config.use_wandb):
        wandb.finish()


if __name__ == "__main__":
    train()
