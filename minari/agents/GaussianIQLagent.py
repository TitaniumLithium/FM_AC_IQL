import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.minari_replaybuffer import ReplayBuffer
from utils.tools import soft_update
import numpy as np
import copy
from typing import Dict, Iterable, List, Tuple
from utils.networks import mlp

class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256), log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dims, 2 * act_dim)
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def dist(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        mean, log_std = self.forward(obs)
        base = torch.distributions.Independent(
            torch.distributions.Normal(mean, log_std.exp()), 1
        )
        return torch.distributions.TransformedDistribution(
            base,
            [torch.distributions.TanhTransform(cache_size=1)],
        )

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -0.999999, 0.999999)
        dist = self.dist(obs)
        return dist.log_prob(action)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, _ = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        return self.sample(obs)[0]
    
class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden_dims, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.v = mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)


# ------------------------------ IQL -----------------------------------


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
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.grad_clip_norm = grad_clip_norm

        self.actor = GaussianActor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.value = ValueNet(obs_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)

    def value_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        with torch.no_grad():
            target_v = self.value(next_obs)
            target_q = rewards + self.gamma * (1.0 - dones) * target_v
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
            weights = torch.clamp(weights, max=100.0)

        log_prob = self.actor.log_prob(obs, actions)
        loss = -(weights * log_prob).mean()
        info = {
            "actor_loss": float(loss.detach().cpu()),
            "adv_mean": float(adv.mean().detach().cpu()),
            "weight_mean": float(weights.mean().detach().cpu()),
            "log_prob_mean": float(log_prob.mean().detach().cpu()),
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
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
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
        act = self.actor.act(obs_n, deterministic=deterministic)
        return act.squeeze(0).cpu().numpy()