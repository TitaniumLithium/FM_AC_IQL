import minari
import torch
import numpy as np
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Tuple

from gymnasium.spaces.dict import Dict as GymDict

@dataclass
class ReplayBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    act_mean: torch.Tensor
    act_std: torch.Tensor

    @property
    def size(self) -> int:
        return self.obs.shape[0]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs.device)
        return {
            "obs": self.normalize_obs(self.obs[idx]),
            "actions": self.actions[idx],
            "next_obs": self.normalize_obs(self.next_obs[idx]),
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_act(self, act: torch.Tensor) -> torch.Tensor:
        return (act - self.act_mean) / self.act_std

    def denormalize_act(self, n_act: torch.Tensor) -> torch.Tensor:
        return self.act_std * n_act +  self.act_mean


@dataclass
class DatasetBundle:
    replay: ReplayBuffer
    env: object
    obs_dim: int
    act_dim: int
    act_low: np.ndarray
    act_high: np.ndarray


def extract_observation(obs):
    """
    Convert Minari observation into a flat state array.
    """

    if isinstance(obs, dict):

        if "observation" in obs:
            return np.asarray(obs["observation"], dtype=np.float32)
            print(f"extracting dict ep.observations[observation] as observation")

        raise ValueError(
            f"Unsupported dict observation keys: {obs.keys()}"
        )

    return np.asarray(obs, dtype=np.float32)

def extract_obs_shape(obs_space):
    if isinstance(obs_space, GymDict):
        res = obs_space.get('observation', None)
        if res is not None:
            print(f"Observation space is a dict, using obs_space['observation'].shape: {res.shape}")
            return res.shape

        raise ValueError(
            f"Unsupported dict observation keys: {obs_space.keys()}"
        )
    
    else:
        print(f"Observation space is not a dict, using obs_space.shape: {obs_space.shape}")
        return obs_space.shape


def load_minari_dataset(dataset_id: str, device: torch.device, recover = True, **kwargs) -> DatasetBundle:
    dataset = minari.load_dataset(dataset_id)
    # Minari docs state this dataset can be recovered from the same env spec;
    # eval_env=True is the intended online evaluation env when available.
    if recover:
        try:
            env = dataset.recover_environment(eval_env=True, **kwargs)
        except Exception:
            env = dataset.recover_environment(**kwargs)
    else:
        env = None

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    rew_list: List[np.ndarray] = []
    done_list: List[np.ndarray] = []

    for ep in dataset.iterate_episodes():
        obs = extract_observation(ep.observations)
        actions = np.asarray(ep.actions, dtype=np.float32)
        rewards = np.asarray(ep.rewards, dtype=np.float32)
        terminations = np.asarray(ep.terminations, dtype=np.bool_)
        truncations = np.asarray(ep.truncations, dtype=np.bool_)
        dones = np.logical_or(terminations, truncations)

        # observations include the initial state, so obs[t] -> obs[t+1]
        obs_list.append(obs[:-1])
        next_obs_list.append(obs[1:])
        act_list.append(actions)
        rew_list.append(rewards[:, None])
        done_list.append(dones[:, None].astype(np.float32))

    obs_arr = np.concatenate(obs_list, axis=0)
    act_arr = np.concatenate(act_list, axis=0)
    next_obs_arr = np.concatenate(next_obs_list, axis=0)
    rew_arr = np.concatenate(rew_list, axis=0)
    done_arr = np.concatenate(done_list, axis=0)

    obs_mean = torch.as_tensor(obs_arr.mean(axis=0), device=device, dtype=torch.float32)
    obs_std = torch.as_tensor(obs_arr.std(axis=0) + 1e-6, device=device, dtype=torch.float32)
    act_mean = torch.as_tensor(act_arr.mean(axis=0), device=device, dtype=torch.float32)
    act_std = torch.as_tensor(act_arr.std(axis=0) + 1e-6, device=device, dtype=torch.float32)

    replay = ReplayBuffer(
        obs=torch.as_tensor(obs_arr, device=device, dtype=torch.float32),
        actions=torch.as_tensor(act_arr, device=device, dtype=torch.float32),
        next_obs=torch.as_tensor(next_obs_arr, device=device, dtype=torch.float32),
        rewards=torch.as_tensor(rew_arr, device=device, dtype=torch.float32),
        dones=torch.as_tensor(done_arr, device=device, dtype=torch.float32),
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
    )

    obs_space = dataset.observation_space
    act_space = dataset.action_space
    assert hasattr(obs_space, "shape") and hasattr(act_space, "shape")

    return DatasetBundle(
        replay=replay,
        env=env,
        obs_dim=int(np.prod(extract_obs_shape(obs_space))),
        act_dim=int(np.prod(act_space.shape)),
        act_low=np.asarray(act_space.low, dtype=np.float32),
        act_high=np.asarray(act_space.high, dtype=np.float32),
    )