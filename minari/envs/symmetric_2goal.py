# symmetric_goal_env_continuous.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _segment_circle_intersect(p1: np.ndarray, p2: np.ndarray, c: np.ndarray, r: float) -> bool:
    """
    判断线段 p1->p2 是否与圆心 c、半径 r 的圆相交。
    """
    d = p2 - p1
    f = p1 - c

    a = float(np.dot(d, d))
    if a == 0.0:
        return float(np.dot(f, f)) <= r * r

    t = -float(np.dot(f, d)) / a
    t = max(0.0, min(1.0, t))
    closest = p1 + t * d
    return float(np.sum((closest - c) ** 2)) <= r * r


@dataclass
class Entity:
    x: float
    y: float
    r: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class SymmetricGoalEnvContinuous(gym.Env):
    """
    连续动作版 2D 平面环境。

    动作:
        action = np.array([dx, dy], dtype=np.float32)
        环境内部会将其归一化为单位方向，再乘以固定 step_size。

    观测:
        [agent_x, agent_y,
         goal1_x, goal1_y,
         goal2_x, goal2_y,
         obs1_x,  obs1_y,
         obs2_x,  obs2_y]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        symmetry: Literal["axis", "center"] = "axis",
        world_limit: float = 10.0,
        step_size: float = 0.6,
        goal_radius: float = 0.7,
        obstacle_radius: float = 0.8,
        max_steps: int = 200,
        goal_reward: float = 10.0,
        step_penalty: float = -0.01,
        invalid_move_penalty: float = -0.2,
        obstacle_penalty: float = -0.5,
        shaping_coef: float = 0.05,
        custom_goals: Optional[list[tuple[float, float]]] = None,
        custom_obstacles: Optional[list[tuple[float, float]]] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.symmetry = symmetry
        self.world_limit = float(world_limit)
        self.step_size = float(step_size)
        self.goal_radius = float(goal_radius)
        self.obstacle_radius = float(obstacle_radius)
        self.max_steps = int(max_steps)

        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.invalid_move_penalty = float(invalid_move_penalty)
        self.obstacle_penalty = float(obstacle_penalty)
        self.shaping_coef = float(shaping_coef)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-self.world_limit,
            high=self.world_limit,
            shape=(10,),
            dtype=np.float32,
        )

        if custom_goals is not None:
            self.goals = [Entity(x, y, self.goal_radius) for x, y in custom_goals]
        else:
            if symmetry == "axis":
                self.goals = [
                    Entity(+6.0, +3.0, self.goal_radius),
                    Entity(-6.0, +3.0, self.goal_radius),
                ]
            elif symmetry == "center":
                self.goals = [
                    Entity(+6.0, +3.0, self.goal_radius),
                    Entity(-6.0, -3.0, self.goal_radius),
                ]
            else:
                raise ValueError(f"Unknown symmetry: {symmetry}")

        if custom_obstacles is not None:
            self.obstacles = [Entity(x, y, self.obstacle_radius) for x, y in custom_obstacles]
        else:
            if symmetry == "axis":
                self.obstacles = [
                    Entity(+3.0, +1.5, self.obstacle_radius),
                    Entity(-3.0, +1.5, self.obstacle_radius),
                ]
            else:
                self.obstacles = [
                    Entity(+3.0, +1.5, self.obstacle_radius),
                    Entity(-3.0, -1.5, self.obstacle_radius),
                ]

        self._agent = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._trajectory: list[np.ndarray] = []

        self._fig = None
        self._ax = None

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._agent[0], self._agent[1],
                self.goals[0].x, self.goals[0].y,
                self.goals[1].x, self.goals[1].y,
                self.obstacles[0].x, self.obstacles[0].y,
                self.obstacles[1].x, self.obstacles[1].y,
            ],
            dtype=np.float32,
        )

    def _min_distance_to_goals(self, pos: np.ndarray) -> float:
        dists = [np.linalg.norm(pos - g.pos) - g.r for g in self.goals]
        return float(np.min(dists))

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {a.shape}")

        norm = float(np.linalg.norm(a))
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float32)

        return a / norm

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self._agent = np.array([0.0, 0.0], dtype=np.float32)
        self._step_count = 0
        self._trajectory = [self._agent.copy()]

        obs = self._get_obs()
        info = {
            "agent_pos": self._agent.copy(),
            "goals": [(g.x, g.y, g.r) for g in self.goals],
            "obstacles": [(o.x, o.y, o.r) for o in self.obstacles],
        }
        return obs, info

    def step(self, action):
        prev_pos = self._agent.copy()
        prev_potential = -self._min_distance_to_goals(prev_pos)

        direction = self._normalize_action(action)
        candidate = prev_pos + self.step_size * direction

        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {
            "invalid_move": False,
            "hit_obstacle": False,
            "reached_goal": False,
            "goal_index": None,
        }

        # 边界裁剪
        clipped = np.clip(candidate, -self.world_limit, self.world_limit)
        if not np.allclose(candidate, clipped):
            candidate = clipped
            reward += self.invalid_move_penalty
            info["invalid_move"] = True

        # 障碍物检测：若穿过障碍，原地不动
        hit_obstacle = False
        for obs in self.obstacles:
            if _segment_circle_intersect(prev_pos, candidate, obs.pos, obs.r):
                hit_obstacle = True
                break

        if hit_obstacle:
            reward += self.obstacle_penalty
            info["hit_obstacle"] = True
            candidate = prev_pos.copy()

        # 目标检测：若进入任一目标圈，则终止
        reached_goal = False
        reached_idx = None
        if not hit_obstacle:
            for i, g in enumerate(self.goals):
                if _segment_circle_intersect(prev_pos, candidate, g.pos, g.r):
                    reached_goal = True
                    reached_idx = i
                    break

        self._agent = candidate
        self._trajectory.append(self._agent.copy())

        curr_potential = -self._min_distance_to_goals(self._agent)
        reward += self.shaping_coef * (curr_potential - prev_potential)

        if reached_goal:
            reward += self.goal_reward
            terminated = True
            info["reached_goal"] = True
            info["goal_index"] = reached_idx

        self._step_count += 1
        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info["agent_pos"] = self._agent.copy()
        info["step_count"] = self._step_count
        info["distance_to_goal"] = self._min_distance_to_goals(self._agent)

        return obs, float(reward), terminated, truncated, info

    def _draw(self):
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._fig.tight_layout()

        ax = self._ax
        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.world_limit, self.world_limit)
        ax.set_ylim(-self.world_limit, self.world_limit)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title("Symmetric Goal Environment (Continuous Action)")

        traj = np.array(self._trajectory, dtype=np.float32)
        if len(traj) >= 2:
            ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.8, label="trajectory")

        ax.scatter(self._agent[0], self._agent[1], s=80, c="black", label="agent", zorder=5)

        for i, g in enumerate(self.goals):
            circle = Circle((g.x, g.y), g.r, color="green", alpha=0.25)
            ax.add_patch(circle)
            ax.scatter([g.x], [g.y], s=40, c="green", zorder=4)
            ax.text(g.x + 0.15, g.y + 0.15, f"goal{i+1}", fontsize=9)

        for i, o in enumerate(self.obstacles):
            circle = Circle((o.x, o.y), o.r, color="red", alpha=0.30)
            ax.add_patch(circle)
            ax.scatter([o.x], [o.y], s=40, c="red", zorder=4)
            ax.text(o.x + 0.15, o.y + 0.15, f"obs{i+1}", fontsize=9)

        ax.legend(loc="upper right")

    def render(self):
        if self.render_mode is None:
            return None

        self._draw()

        if self.render_mode == "human":
            plt.show(block=False)
            plt.pause(0.001)
            return None

        if self.render_mode == "rgb_array":
            canvas = FigureCanvas(self._fig)
            canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape(h, w, 4)[..., :3].copy()
            return img

        raise ValueError(f"Unsupported render_mode: {self.render_mode}")

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None


if __name__ == "__main__":
    env = SymmetricGoalEnvContinuous(render_mode="human", symmetry="axis")
    obs, info = env.reset(seed=42)

    terminated = truncated = False
    while not (terminated or truncated):
        # 示例：随机连续动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()