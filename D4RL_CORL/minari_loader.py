import minari

import torch

import numpy as np
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Tuple

from envs.pusht import PushTEnv

def load_env_minari(dataset_id: str, **kwargs):
    dataset = minari.load_dataset(dataset_id)
    # Minari docs state this dataset can be recovered from the same env spec;
    # eval_env=True is the intended online evaluation env when available.
    if "pusht" not in dataset_id:
        try:
            env = dataset.recover_environment(eval_env=True, **kwargs)
        except Exception:
            env = dataset.recover_environment(**kwargs)
    else:
        env = PushTEnv(render_mode="rgb_array")
    return env,dataset

def load_minari_dataset(dataset_id: str, device: torch.device, **kwargs):
    dataset = minari.load_dataset(dataset_id)
    # Minari docs state this dataset can be recovered from the same env spec;
    # eval_env=True is the intended online evaluation env when available.
    if "pusht" not in dataset_id:
        try:
            env = dataset.recover_environment(eval_env=True, **kwargs)
        except Exception:
            env = dataset.recover_environment(**kwargs)
    else:
        env = PushTEnv(render_mode="rgb_array")

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    rew_list: List[np.ndarray] = []
    done_list: List[np.ndarray] = [] 
    terminal_list: List[np.ndarray] = []

    for ep in dataset.iterate_episodes():
        obs = np.asarray(ep.observations, dtype=np.float32)
        actions = np.asarray(ep.actions, dtype=np.float32)
        rewards = np.asarray(ep.rewards, dtype=np.float32)
        terminations = np.asarray(ep.terminations, dtype=np.bool_)
        truncations = np.asarray(ep.truncations, dtype=np.bool_)
        dones = np.logical_or(terminations, truncations)

        # observations include the initial state, so obs[t] -> obs[t+1]
        obs_list.append(obs[:-1])
        next_obs_list.append(obs[1:])
        act_list.append(actions)
        rew_list.append(rewards)
        done_list.append(dones.astype(np.float32))
        terminal_list.append(terminations.astype(np.float32))

    obs_arr = np.concatenate(obs_list, axis=0)
    act_arr = np.concatenate(act_list, axis=0)
    next_obs_arr = np.concatenate(next_obs_list, axis=0)
    rew_arr = np.concatenate(rew_list, axis=0).flatten()
    done_arr = np.concatenate(done_list, axis=0).flatten()
    terminal_arr = np.concatenate(terminal_list, axis=0).flatten()

    assert obs_arr.shape[0] ==  act_arr.shape[0] == next_obs_arr.shape[0] == rew_arr.shape[0] == terminal_arr.shape[0]

    print(obs_arr.shape,act_arr.shape,done_arr.shape)

    data = {}

    data["observations"] = obs_arr
    data["actions"] = act_arr
    data["rewards"] = rew_arr
    data["next_observations"] = next_obs_arr
    data["terminals"] = terminal_arr
    data["timeouts"] = done_arr

    return data,env
