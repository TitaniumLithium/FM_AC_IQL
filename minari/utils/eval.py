from typing import Dict, Iterable, List, Tuple
import numpy as np
import imageio
import os
import torch
from utils.minari_chunkreplaybuffer import extract_observation

def get_fps(env, default=30):
    if hasattr(env.unwrapped, "dt") and env.unwrapped.dt:
        return int(round(1.0 / env.unwrapped.dt))
    return int(env.metadata.get("render_fps", default))


@torch.no_grad()
def evaluate_policy(
    agent,
    env,
    replay,
    episodes: int,
    seed: int,
    max_steps:int = 1000,
    use_ema = True,
    chunk_len = 4,
) -> Dict[str, float]:
    returns: List[float] = []
    lengths: List[int] = []
    infos = []
    if use_ema:
        agent.ema_copy()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = extract_observation(obs)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            action_chunk = agent.act(obs, replay=replay, deterministic=True) # [H,act_dim]
            L = min(chunk_len,action_chunk.shape[0])
            for h in range(L):
                action = action_chunk[h,:]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = extract_observation(obs)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1
                if done:
                    break
            if ep_len>max_steps:
                break
        
        infos.append(info)

        returns.append(ep_ret)
        lengths.append(ep_len)
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "infos": infos
    }

@torch.no_grad()
def evaluate_policy_video(
    agent,
    env,
    replay,
    seed,
    save_dir,
    max_steps:int = 1000,
    use_ema = True,
    chunk_len = 4,
) -> Dict[str, float]:
    episodes = 1
    returns: List[float] = []
    lengths: List[int] = []
    if use_ema:
        agent.ema_copy()
    writer = imageio.get_writer(
    save_dir,
    fps=get_fps(env),
    codec="libx264"
    )
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = extract_observation(obs)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            action_chunk = agent.act(obs, replay=replay, deterministic=True) # [H,act_dim]
            L = min(chunk_len,action_chunk.shape[0])
            for h in range(L):
                frame = env.render()
                writer.append_data(frame)
                action = action_chunk[h,:]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = extract_observation(obs)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1
                if done:
                    break
            if ep_len>max_steps:
                break

        returns.append(ep_ret)
        lengths.append(ep_len)

    writer.close()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }