import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import minari
import os
import gymnasium as gym
from typing import Dict, Iterable, List, Tuple

from agents.GaussianIQLagent import IQLAgent
from utils.tools import set_seed,soft_update,to_tensor
from utils.minari_replaybuffer import load_minari_dataset,ReplayBuffer


@torch.no_grad()
def evaluate_policy(
    agent: IQLAgent,
    env,
    replay: ReplayBuffer,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    returns: List[float] = []
    lengths: List[int] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            action = agent.act(obs, replay=replay, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }

def train_agent(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset: {args.dataset_id}")

    bundle = load_minari_dataset(args.dataset_id, device=device)
    replay = bundle.replay
    env = bundle.env

    print(
        f"Dataset loaded: size={replay.size:,}, obs_dim={bundle.obs_dim}, act_dim={bundle.act_dim}, "
        f"obs_mean_shape={tuple(replay.obs_mean.shape)}"
    )
    
    last_path = args.save_path + "iql_last.pt"
    best_path = args.save_path + "iql_best.pt"

    agent = IQLAgent(
        obs_dim=bundle.obs_dim,
        act_dim=bundle.act_dim,
        device=device,
        gamma=args.gamma,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        grad_clip_norm=args.grad_clip_norm,
    )
    
    save_videos = bool(args.save_videos and args.env_id is not None)
    
    use_wandb = bool(args.use_wandb and wandb is not None)
    if args.use_wandb and wandb is None:
        print("wandb is not installed; continuing without wandb logging.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or None,
            config=vars(args),
        )
        wandb.watch(agent.actor, log="gradients", log_freq=max(1, args.log_interval))
        
    num_epochs = args.num_steps//args.batch_size + 1
    if args.batch_size > args.log_interval:
        args.log_interval = args.batch_size
    if args.batch_size > args.eval_interval:
        args.eval_interval = args.batch_size
    global_steps = 0
    next_log_step = args.log_interval
    next_eval_step = args.eval_interval
    
    print(f"num_epochs = {args.num_steps}//{args.batch_size} + 1 = {num_epochs}")
    
    best_eval = -float("inf")
    pbar = tqdm(range(1, num_epochs + 1), desc="training", dynamic_ncols=True)
    
    for epoch in pbar:
        batch = replay.sample(args.batch_size)
        metrics = agent.update(batch, replay)
        global_steps += args.batch_size

        if global_steps >= next_log_step:
            next_log_step += args.log_interval
            pbar.set_postfix(
                {
                    "v": f"{metrics['value_loss']:.3f}",
                    "q": f"{metrics['critic_loss']:.3f}",
                    "a": f"{metrics['actor_loss']:.3f}",
                    "ret": f"{best_eval:.1f}",
                }
            )
            print(
                f"step={global_steps:>7d} "
                f"loss_total={metrics['loss_total']:.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"adv_mean={metrics['adv_mean']:.4f} "
                f"weight_mean={metrics['weight_mean']:.4f} "
                f"q1_mean={metrics['q1_mean']:.4f} q2_mean={metrics['q2_mean']:.4f} "
                f"v_mean={metrics['v_mean']:.4f}"
            )
            if use_wandb:
                wandb.log({**metrics, "global_steps": global_steps, "epoch": epoch}, step=epoch)

        if global_steps >= next_eval_step:
            next_eval_step += args.eval_interval
            eval_metrics = evaluate_policy(agent, env, replay, episodes=args.eval_episodes, seed=args.seed + 1000)
            print(
                f"[EVAL] step={global_steps:>7d} "
                f"return_mean={eval_metrics['eval_return_mean']:.2f} ± {eval_metrics['eval_return_std']:.2f} "
                f"len_mean={eval_metrics['eval_length_mean']:.1f}"
            )
            if use_wandb:
                wandb.log({**eval_metrics, "global_steps": global_steps, "epoch": epoch}, step=epoch)

            if eval_metrics["eval_return_mean"] > best_eval:
                best_eval = eval_metrics["eval_return_mean"]
                ckpt = {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "critic_target": agent.critic_target.state_dict(),
                    "value": agent.value.state_dict(),
                    "obs_mean": replay.obs_mean,
                    "obs_std": replay.obs_std,
                    "args": vars(args),
                    "best_eval_return_mean": best_eval,
                }
                torch.save(ckpt, best_path)
                print(f"Saved best checkpoint to {best_path}")

    if use_wandb:
        wandb.finish()