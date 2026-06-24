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
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden_dims, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        obs [B, obs_dim]
        action [B, act_dim]
        '''
        x = torch.cat([obs, action], dim=-1)
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
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = FMActor(obs_dim, act_dim, device,act_horizon=act_horizon, chunk_len=chunk_len, unet_dims=unet_dims, cond_dim = cond_dim, time_emb_dim = time_emb_dim, dropout=dropout)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.value = ValueNet(obs_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.net.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        self.ema = EMA(self.actor.net, decay=ema_decay) if use_ema else None

        self.gamma_powers = (self.gamma ** torch.arange(self.act_horizon, dtype=torch.float32))[None, :, None].to(device)

    def value_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''
        obs [B, obs_dim]
        actions [B, act_dim]
        '''
        with torch.no_grad():
            q1_t, q2_t = self.critic_target(obs, actions)
            q = torch.min(q1_t, q2_t)
        v = self.value(obs)
        diff = q - v
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        loss = (weight * diff.square()).mean()
        info = {
            "value_loss": loss.detach(),
            "v_mean": v.mean().detach(),
            "q_target_mean": q.mean().detach(),
            "expectile_weight_mean": weight.mean().detach(),
        }
        return loss, info

    def critic_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            target_v = self.value(next_obs)
            target_q = rewards + self.gamma * (1.0 - dones) * target_v.detach()
        q1, q2 = self.critic(obs, actions)
        loss1 = F.mse_loss(q1, target_q)
        loss2 = F.mse_loss(q2, target_q)
        loss = loss1 + loss2
        info = {
            "critic_loss": loss.detach(),
            "q1_mean": q1.mean().detach(),
            "q2_mean": q2.mean().detach(),
            "target_q_mean": target_q.mean().detach(),
        }
        return loss, info

    def actor_loss(self, obs_chunk: torch.Tensor, action_chunk: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        init_obs = obs_chunk[:, 0, :]
        with torch.no_grad():
            q1, q2 = self.critic_target(obs_chunk, action_chunk)
            q = torch.min(q1, q2)
            v = self.value(obs_chunk)
            adv = q - v
            dist_adv = adv * self.gamma_powers # [B,H,1]
            dist_adv = dist_adv.squeeze()
            chunk_adv = dist_adv.mean(dim=-1, keepdim=True)
            # IQL paper uses advantage-weighted regression. In the official JAX code,
            # temperature is a multiplier on advantage, so larger values sharpen the weights.
            weights = torch.exp(torch.clamp(chunk_adv.detach() * self.temperature, max=100.0))
            weights = torch.clamp(weights, max=100.0) #[B,1]

        fm_loss = self.actor.flow_match_loss(init_obs,action_chunk) # [B]
        fm_loss = fm_loss.reshape(-1,1)

        loss = (weights * fm_loss).mean()
        info = {
            "actor_loss": loss.detach(),
            "adv_mean": chunk_adv.mean().detach(),
            "weight_mean": weights.mean().detach(),
        }
        return loss, info

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}

        # 1) value update
        self.value_opt.zero_grad(set_to_none=True)
        v_loss, v_info = self.value_loss(batch["obs"].reshape(-1,self.obs_dim), batch["actions"].reshape(-1,self.act_dim))
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
            batch["obs"].reshape(-1,self.obs_dim), batch["actions"].reshape(-1,self.act_dim), batch["next_obs"].reshape(-1,self.obs_dim), batch["rewards"].reshape(-1,1), batch["dones"].reshape(-1,1)
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
                "loss_total": (v_loss + a_loss + c_loss).detach(),
                "critic_q_gap": abs(metrics["q1_mean"] - metrics["q2_mean"]),
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