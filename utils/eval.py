from typing import Dict, Iterable, List, Tuple
import numpy as np
import imageio
import os
import torch


@torch.no_grad()
def evaluate_policy(
    agent,
    env,
    replay,
    episodes: int,
    seed: int,
    max_steps:int = 1000
) -> Dict[str, float]:
    returns: List[float] = []
    lengths: List[int] = []
    agent.ema_copy()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            action_chunk = agent.act(obs, replay=replay, deterministic=True) # [H,act_dim]
            for h in range(action_chunk.shape[0]):
                action = action_chunk[h,:]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1
                if done:
                    break
            if ep_len>max_steps:
                break

        returns.append(ep_ret)
        lengths.append(ep_len)
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }

@torch.no_grad()
def evaluate_policy_video(
    agent,
    env,
    replay,
    seed,
    save_dir,
    max_steps:int = 1000
) -> Dict[str, float]:
    episodes = 1
    returns: List[float] = []
    lengths: List[int] = []
    agent.ema_copy()
    writer = imageio.get_writer(
    save_dir,
    fps=int(1 / env.unwrapped.dt),
    codec="libx264"
    )
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            action_chunk = agent.act(obs, replay=replay, deterministic=True) # [H,act_dim]
            for h in range(action_chunk.shape[0]):
                frame = env.render()
                writer.append_data(frame)
                action = action_chunk[h,:]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
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