import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.networks import EMA,FM1DUnet,mlp
from typing import Dict, Iterable, List, Tuple
from utils.minari_chunkreplaybuffer import ReplayBuffer
from utils.tools import soft_update
import numpy as np
import copy


class FMActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, act_horizon=4, chunk_len=4, unet_dims=[128, 256, 512], cond_dim = 128, time_emb_dim = 128, dropout=0.1):
        super().__init__()
        self.act_horizon = min(4,act_horizon)
        self.net = FM1DUnet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            horizon=self.act_horizon,
            down_dims=unet_dims,
            kernel_size=3,
            cond_dim=cond_dim,
            time_emb_dim=time_emb_dim,
            dropout=0.1,
        ).to(device)
        self.shadow_model = FM1DUnet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            horizon=self.act_horizon,
            down_dims=unet_dims,
            kernel_size=3,
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
    ) -> Tuple[torch.Tensor, dict]:
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
        pred_velocity = model(obs_seq, x_tau, tau)

        err = (pred_velocity - target_velocity)**2
        loss = err.mean(dim=(1,2))   # [B]

        metrics = {
            "loss": loss.detach()
        }
        return loss, metrics

    @torch.no_grad()
    def sample_action_chunk(
        self,
        obs: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Iterative denoising / ODE-style sampling from noise to action chunk."""
        device = self.device
        model = self.shadow_model
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


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, chunk_len: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim * chunk_len, hidden_dims, 1)
        self.q2 = mlp(obs_dim + act_dim * chunk_len, hidden_dims, 1)
        self.chunk_dim = act_dim * chunk_len

    def forward(self, obs: torch.Tensor, action_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        obs [B, obs_dim]
        action_chunk [B, H, act_dim]
        '''
        act = action_chunk.reshape(-1,self.chunk_dim)
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.v = mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)


class IQLAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        grad_clip_norm: float = 1.0,
        act_horizon: int = 4,
        chunk_len: int = 4,
        unet_dims=[128, 256, 512],
        cond_dim = 128,
        time_emb_dim =128,
        dropout=0.1,
        ema_decay = 0.999,
        use_ema = True
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.grad_clip_norm = grad_clip_norm
        self.act_horizon = act_horizon
        self.chunk_len = chunk_len

        self.actor = FMActor(obs_dim, act_dim, device,act_horizon=act_horizon, chunk_len=chunk_len, unet_dims=unet_dims, cond_dim = cond_dim, time_emb_dim = time_emb_dim, dropout=dropout)
        self.critic = Critic(obs_dim, act_dim, chunk_len).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.value = ValueNet(obs_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.net.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        self.ema = EMA(self.actor.net, decay=ema_decay) if use_ema else None

    def value_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        '''
        obs [B, obs_dim]
        actions [B, H, act_dim]
        '''
        with torch.no_grad():
            q1_t, q2_t = self.critic_target(obs, actions)
            q = torch.min(q1_t, q2_t)
        v = self.value(obs)
        diff = q - v
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        loss = (weight * diff.square()).mean()
        info = {
            "value_loss": float(loss.detach().cpu()),
            "v_mean": float(v.mean().detach().cpu()),
            "q_target_mean": float(q.mean().detach().cpu()),
            "expectile_weight_mean": float(weight.mean().detach().cpu()),
        }
        return loss, info

    def critic_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        chunk_gamma = self.gamma ** self.chunk_len
        with torch.no_grad():
            target_v = self.value(next_obs)
            target_q = rewards + chunk_gamma * (1.0 - dones) * target_v
        q1, q2 = self.critic(obs, actions)
        loss1 = F.mse_loss(q1, target_q)
        loss2 = F.mse_loss(q2, target_q)
        loss = loss1 + loss2
        info = {
            "critic_loss": float(loss.detach().cpu()),
            "q1_mean": float(q1.mean().detach().cpu()),
            "q2_mean": float(q2.mean().detach().cpu()),
            "target_q_mean": float(target_q.mean().detach().cpu()),
        }
        return loss, info

    def actor_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            q1, q2 = self.critic_target(obs, actions)
            q = torch.min(q1, q2)
            v = self.value(obs)
            adv = q - v
            # IQL paper uses advantage-weighted regression. In the official JAX code,
            # temperature is a multiplier on advantage, so larger values sharpen the weights.
            weights = torch.exp(torch.clamp(adv * self.temperature, max=100.0))
            weights = torch.clamp(weights, max=100.0) #[B,1]

        fm_loss, _ = self.actor.flow_match_loss(obs,actions) # [B]
        fm_loss = fm_loss.reshape(-1,1)

        loss = (weights * fm_loss).mean()
        info = {
            "actor_loss": float(loss.detach().cpu()),
            "adv_mean": float(adv.mean().detach().cpu()),
            "weight_mean": float(weights.mean().detach().cpu()),
        }
        return loss, info

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # 1) value update
        self.value_opt.zero_grad(set_to_none=True)
        v_loss, v_info = self.value_loss(batch["obs"], batch["actions"])
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip_norm)
        self.value_opt.step()
        metrics.update(v_info)

        # 2) actor update (uses updated value network)
        self.actor_opt.zero_grad(set_to_none=True)
        a_loss, a_info = self.actor_loss(batch["obs"], batch["actions"])
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.grad_clip_norm)
        self.actor_opt.step()
        metrics.update(a_info)

        # 3) critic update
        self.critic_opt.zero_grad(set_to_none=True)
        c_loss, c_info = self.critic_loss(
            batch["obs"], batch["actions"], batch["next_obs"], batch["rewards"], batch["dones"]
        )
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_opt.step()
        metrics.update(c_info)

        # 4) target critic update
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.ema.update(self.actor.net)

        metrics.update(
            {
                "loss_total": float((v_loss + a_loss + c_loss).detach().cpu()),
                "critic_q_gap": float(abs(metrics["q1_mean"] - metrics["q2_mean"])),
            }
        )
        return metrics

    @torch.no_grad()
    def act(self, obs: np.ndarray, replay: ReplayBuffer, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        obs_n = replay.normalize_obs(obs_t)
        n_act = self.actor.sample_action_chunk(obs_n)
        act = replay.denormalize_act(n_act)
        return act.squeeze(0).cpu().numpy()
    
    def ema_copy(self):
        self.actor.shadow_model.load_state_dict(self.actor.net.state_dict())
        self.ema.copy_to(self.actor.shadow_model)
        self.actor.shadow_model.eval()