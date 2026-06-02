import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import minari
import os
import gymnasium as gym

from agents.FMIQLagent import IQLAgent
from utils.eval import evaluate_policy,evaluate_policy_video
from utils.tools import set_seed,soft_update
from utils.minari_chunkreplaybuffer import load_minari_dataset

def train_agent(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset: {args.dataset_id}")

    bundle = load_minari_dataset(args.dataset_id, device=device)
    replay = bundle.replay
    env = bundle.env

    print(
        f"Dataset loaded: size={replay.size:,}, obs_dim={bundle.obs_dim}, act_dim={bundle.act_dim}, "
        f"obs_mean_shape={tuple(replay.obs_mean.shape)} "
        f"normalized: obs {replay.obs_mean.mean()}+-{replay.obs_std.mean()} act {replay.act_mean.mean()}+-{replay.act_std.mean()}"
    )

    last_path = args.save_path + "last.pt"
    best_path = args.save_path + "best.pt"

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
        obs_horizon=1,
        act_horizon=args.chunk_len,
        chunk_len=args.chunk_len
    )

    use_wandb = bool(args.use_wandb and wandb is not None)
    save_videos = bool(args.save_videos and args.env_id is not None)

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
    step = 0
    next_log_step = args.log_interval
    next_eval_step = args.eval_interval

    print(f"num_epochs = {args.num_steps}//{args.batch_size} + 1 = {num_epochs}")

    best_eval = -float("inf")
    pbar = tqdm(range(1, num_epochs + 1), desc="training", dynamic_ncols=True)

    for epoch in pbar:
        batch = replay.sample(args.batch_size)
        metrics = agent.update(batch)
        step += args.batch_size

        if step >= next_log_step:
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
                f"step={step:>7d} "
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
                wandb.log({**metrics, "step": step, "epoch": epoch}, step=epoch)


        if step >= next_eval_step:
            next_eval_step += args.eval_interval
            eval_metrics = evaluate_policy(agent, env, replay, episodes=args.eval_episodes, seed=args.seed + 1000)
            print(
                f"[EVAL] step={step:>7d} "
                f"return_mean={eval_metrics['eval_return_mean']:.2f} ± {eval_metrics['eval_return_std']:.2f} "
                f"len_mean={eval_metrics['eval_length_mean']:.1f}"
            )
            last_path = args.save_path + f"last_{step}.pt"

            ckpt = {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "value": agent.value.state_dict(),
                "obs_mean": replay.obs_mean,
                "obs_std": replay.obs_std,
                "act_mean": replay.act_mean,
                "act_std": replay.act_std,
                "args": vars(args),
                "best_eval_return_mean": best_eval,
            }

            torch.save(ckpt, last_path)

            if use_wandb:
                wandb.log({**eval_metrics, "step": step, "epoch": epoch}, step=epoch)
                artifact_last = wandb.Artifact(
                    name=f"last_agent_{step}",
                    type="model"
                )
                artifact_last.add_file(last_path)
                wandb.log_artifact(artifact_last)
            

            if eval_metrics["eval_return_mean"] > best_eval:
                best_eval = eval_metrics["eval_return_mean"]
                torch.save(ckpt, best_path)
                print(f"Saved best checkpoint to {best_path}")
                if save_videos:
                    video_path = "./videos/" + f"rollout_{step}.mp4"
                    video_eval = gym.make(args.env_id,render_mode="rgb_array")
                    evaluate_policy_video(agent, video_eval, replay, args.seed + 1000, video_path)
                if use_wandb:
                    artifact_best = wandb.Artifact(
                        name="best_agent",
                        type="model"
                    )
                    artifact_best.add_file(best_path)
                    wandb.log_artifact(artifact_best)
                    if save_videos:
                        wandb.log({
                            "video": wandb.Video(
                                video_path,
                                fps=30,
                                format="mp4"
                            )
                        })

    if use_wandb:
        wandb.finish()