import torch
import gymnasium as gym
from envs.symmetric_2goal import SymmetricGoalEnvContinuous
import numpy as np

from agents.FMIQLagent import IQLAgent,FMActor
from utils.minari_chunkreplaybuffer import load_minari_dataset

ckpt_path = f"checkpoints/pusht_best.pt"

'''
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

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device={device}")

ckpt = torch.load(ckpt_path,weights_only=False,map_location=device)

actor = FMActor(
    obs_dim= 10,
    act_dim= 2, 
    device= device, 
    act_horizon=4, 
    chunk_len=4, 
    unet_dims=[128, 256], 
    cond_dim = 128, 
    time_emb_dim = 128, 
    dropout=0.1
).to(device)

actor.load_state_dict(ckpt["actor"])

env = SymmetricGoalEnvContinuous(render_mode="rgb_array")

seed = 0

obs, _ = env.reset(seed=seed)

bundle = load_minari_dataset("symmetric_goal/human_demo-v0", device=device,recover=False,horizon=4)
replay = bundle.replay

@torch.no_grad()
def act(obs, replay, device, actor) -> np.ndarray:
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    obs_n = replay.normalize_obs(obs_t)
    n_act = actor.sample_action_chunk(obs_n)
    act = replay.denormalize_act(n_act)
    return act.squeeze(0).cpu().numpy()

action_chunk = act(obs, replay, device, actor)

print(action_chunk)

n_samples = 1000
act_seq_idx = 0
    
max_length = 50

data = np.zeros((n_samples,4,2))

print(data.shape)

data_idx = 0

done = False
ep_len = 0
obs, _ = env.reset(seed=seed)

while not done:
    action_chunk = act(obs, replay, device, actor)
    for h in range(4):
        action = action_chunk[h,:]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        ep_len += 1
        if done:
            break
            
    if ep_len>max_length:
        break
            
print("is success: ",info["reached_goal"]," ep_len ",ep_len)

done = False
ep_len = 0
obs, _ = env.reset(seed=seed)

while not done:
    for i in range(n_samples):
        new_action_chunk = act(obs, replay, device, actor) #[4,act_dim] []
        data[i] = new_action_chunk
        
    data_idx += 1
    np.savetxt(
        f"data_{data_idx}.csv",
        data.reshape(n_samples,-1),
        delimiter=",",
        fmt="%.6f"
    )
        
    for h in range(4):
        action = new_action_chunk[h,:]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        ep_len += 1
        if done:
            break
            
    if ep_len>max_length:
        break
    
    print(ep_len)
            
print("is success: ",info["reached_goal"]," ep_len ",ep_len)
