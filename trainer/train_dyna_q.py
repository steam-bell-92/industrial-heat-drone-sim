from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

State = Tuple[int, int, int]
Action = int


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None and value.strip() else default


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    return Path(value).expanduser().resolve()


@dataclass
class Config:
    grid_x: int = 40
    grid_z: int = 40
    t_bins: int = 12
    n_actions: int = 5
    episodes: int = 8000
    max_steps: int = 420
    alpha: float = 0.18
    gamma: float = 0.985
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.991
    planning_steps: int = 50
    n_heat_zones: int = 24
    warmup_episodes: int = 70
    shaping_scale: float = 0.16
    heat_bonus: float = 2.8
    train_seconds: int = 7200
    log_every_seconds: int = 300
    dist_penalty_scale: float = 0.055
    thermal_penalty_scale: float = 0.012
    idle_penalty: float = 0.02


class IndustrialHeatEnv:
    """Approximate environment with drifting heat sources and noisy thermal sensing."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.heat_centers = [
            (
                random.randint(1, cfg.grid_x - 2),
                random.randint(1, cfg.grid_z - 2),
            )
            for _ in range(cfg.n_heat_zones)
        ]
        self.reset()

    def reset(self) -> State:
        self.step_idx = 0
        if random.random() < 0.5:
            self.x = random.randint(self.cfg.grid_x // 4, (self.cfg.grid_x * 3) // 4)
            self.z = random.randint(self.cfg.grid_z // 4, (self.cfg.grid_z * 3) // 4)
        else:
            self.x = random.randint(0, self.cfg.grid_x - 1)
            self.z = random.randint(0, self.cfg.grid_z - 1)
        return self._observe_state()

    def nearest_hotspot_distance(self, x: int, z: int) -> float:
        t = self.step_idx * 0.08
        return min(
            math.sqrt((x - (cx + 0.8 * math.sin(t * 0.9 + i))) ** 2 + (z - (cz + 0.8 * math.cos(t * 0.7 + i))) ** 2)
            for i, (cx, cz) in enumerate(self.heat_centers)
        )

    def expert_action(self, x: int, z: int) -> Action:
        t = self.step_idx * 0.08
        best_idx = 0
        best_temp = -1.0
        probes = [
            (x + 1, z),
            (x - 1, z),
            (x, z + 1),
            (x, z - 1),
            (x, z),
        ]
        for idx, (px, pz) in enumerate(probes):
            px = min(self.cfg.grid_x - 1, max(0, px))
            pz = min(self.cfg.grid_z - 1, max(0, pz))
            temp = self._thermal_signal(px, pz)
            if temp > best_temp:
                best_temp = temp
                best_idx = idx
        return best_idx

    def _thermal_signal(self, x: int, z: int) -> float:
        t = self.step_idx * 0.08
        total = 18 + 2.5 * math.sin(t)
        for i, (cx, cz) in enumerate(self.heat_centers):
            drift_x = cx + 0.8 * math.sin(t * 0.9 + i)
            drift_z = cz + 0.8 * math.cos(t * 0.7 + i)
            dist2 = (x - drift_x) ** 2 + (z - drift_z) ** 2
            amp = 68 + 24 * math.sin(t * 1.6 + i)
            total += amp * math.exp(-dist2 / 14.0)
        total += random.uniform(-2.2, 2.2)
        return max(10.0, total)

    def _observe_state(self) -> State:
        thermal = self._thermal_signal(self.x, self.z)
        t_bin = min(self.cfg.t_bins - 1, max(0, int(thermal // 15)))
        return (self.x, self.z, t_bin)

    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        prev_dist = self.nearest_hotspot_distance(self.x, self.z)
        self.step_idx += 1
        prev_x, prev_z = self.x, self.z

        if action == 0:
            self.x += 1
        elif action == 1:
            self.x -= 1
        elif action == 2:
            self.z += 1
        elif action == 3:
            self.z -= 1

        self.x = min(self.cfg.grid_x - 1, max(0, self.x))
        self.z = min(self.cfg.grid_z - 1, max(0, self.z))

        s_next = self._observe_state()
        heat_hit = s_next[2] >= 7
        next_dist = self.nearest_hotspot_distance(self.x, self.z)
        moved = (self.x != prev_x) or (self.z != prev_z)

        # Non-positive squared penalties: distance and thermal mismatch are penalized,
        # while a heat-zone hit is the only zero-reward terminal-like signal.
        dist_norm = next_dist / max(1.0, math.sqrt(self.cfg.grid_x * self.cfg.grid_x + self.cfg.grid_z * self.cfg.grid_z))
        target_bin = self.cfg.t_bins - 1
        thermal_gap = max(0.0, (target_bin - s_next[2]) / max(1, self.cfg.t_bins - 1))
        dist_penalty = self.cfg.dist_penalty_scale * (dist_norm**2)
        thermal_penalty = self.cfg.thermal_penalty_scale * (thermal_gap**2)
        move_penalty = 0.0 if moved else self.cfg.idle_penalty

        reward = 0.0 if heat_hit else -(dist_penalty + thermal_penalty + move_penalty)
        done = self.step_idx >= self.cfg.max_steps

        return s_next, reward, done, {"heat_hit": heat_hit, "target_distance": next_dist}


class DynaQAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.q = np.zeros((cfg.grid_x, cfg.grid_z, cfg.t_bins, cfg.n_actions), dtype=np.float32)
        self.model: Dict[Tuple[State, Action], Tuple[State, float]] = {}

    def choose_action(self, state: State, epsilon: float) -> Action:
        if random.random() < epsilon:
            return random.randrange(self.cfg.n_actions)
        x, z, tb = state
        return int(np.argmax(self.q[x, z, tb]))

    def update_q(self, s: State, a: Action, r: float, s_next: State) -> None:
        x, z, tb = s
        nx, nz, ntb = s_next
        td_target = r + self.cfg.gamma * float(np.max(self.q[nx, nz, ntb]))
        td_error = td_target - float(self.q[x, z, tb, a])
        self.q[x, z, tb, a] += self.cfg.alpha * td_error

    def update_model(self, s: State, a: Action, s_next: State, r: float) -> None:
        self.model[(s, a)] = (s_next, r)

    def planning(self) -> None:
        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.cfg.planning_steps):
            s, a = random.choice(keys)
            s_next, r = self.model[(s, a)]
            self.update_q(s, a, r, s_next)


def seeded_curriculum_experience(agent: DynaQAgent, env: IndustrialHeatEnv, passes: int) -> int:
    loaded = 0
    for _ in range(passes):
        s = env.reset()
        for _ in range(env.cfg.max_steps // 2):
            a = env.expert_action(env.x, env.z)
            s_next, r, done, _ = env.step(a)
            agent.update_q(s, a, r, s_next)
            agent.update_model(s, a, s_next, r)
            s = s_next
            loaded += 1
            if done:
                break
    return loaded


def maybe_load_offline_experience(agent: DynaQAgent, path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    loaded = 0
    for ep in payload.get('episodes', []):
        for step in ep.get('steps', []):
            s = tuple(step['state'])
            a = int(step['action'])
            r = float(step['reward'])
            s_next = tuple(step['next_state'])
            if len(s) == 3 and len(s_next) == 3:
                s_clamped: State = (
                    int(np.clip(s[0], 0, agent.cfg.grid_x - 1)),
                    int(np.clip(s[1], 0, agent.cfg.grid_z - 1)),
                    int(np.clip(s[2], 0, agent.cfg.t_bins - 1)),
                )
                s_next_clamped: State = (
                    int(np.clip(s_next[0], 0, agent.cfg.grid_x - 1)),
                    int(np.clip(s_next[1], 0, agent.cfg.grid_z - 1)),
                    int(np.clip(s_next[2], 0, agent.cfg.t_bins - 1)),
                )
                a = int(np.clip(a, 0, agent.cfg.n_actions - 1))
                agent.update_q(s_clamped, a, r, s_next_clamped)
                agent.update_model(s_clamped, a, s_next_clamped, r)
                loaded += 1
    return loaded


def save_policy(agent: DynaQAgent, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_actions = np.argmax(agent.q, axis=3)
    payload = {
        'shape': list(best_actions.shape),
        'best_actions': best_actions.tolist(),
        'q_values': agent.q.tolist(),
    }
    out_path.write_text(json.dumps(payload), encoding='utf-8')


def maybe_load_policy(agent: DynaQAgent, path: Path) -> bool:
    if not path.exists():
        return False
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    q_values = payload.get('q_values')
    if not q_values:
        return False

    q_array = np.array(q_values, dtype=np.float32)
    if q_array.shape != agent.q.shape:
        return False

    agent.q[:] = q_array
    return True


def train(cfg: Config, experience_path: Path) -> None:
    env = IndustrialHeatEnv(cfg)
    agent = DynaQAgent(cfg)

    policy_path = Path(__file__).resolve().parent / 'models' / 'dyna_q_policy.json'
    output_policy_path = _env_path('DYNA_POLICY_OUTPUT_PATH') or policy_path
    if maybe_load_policy(agent, policy_path):
        print(f'Resumed from existing policy: {policy_path}')

    preloaded = maybe_load_offline_experience(agent, experience_path)
    print(f'Loaded offline samples: {preloaded}')

    warm_start = seeded_curriculum_experience(agent, env, cfg.warmup_episodes)
    print(f'Loaded expert warm-start samples: {warm_start}')

    epsilon = cfg.epsilon
    rewards: List[float] = []
    heat_counts: List[int] = []
    target_distances: List[float] = []
    start_time = time.time()
    last_log_time = start_time

    for episode in range(cfg.episodes):
        elapsed = time.time() - start_time
        if elapsed >= cfg.train_seconds:
            print(f'Time budget reached after {elapsed / 60.0:.1f} minutes')
            break

        s = env.reset()
        ep_reward = 0.0
        hits = 0

        if episode < cfg.episodes // 3:
            epsilon_episode = max(0.15, epsilon)
        elif episode < (cfg.episodes * 2) // 3:
            epsilon_episode = max(0.08, epsilon)
        else:
            epsilon_episode = epsilon

        for _ in range(cfg.max_steps):
            expert_bias = 0.7 if episode < cfg.warmup_episodes else 0.35
            if random.random() < expert_bias:
                a = env.expert_action(env.x, env.z)
            else:
                a = agent.choose_action(s, epsilon_episode)
            s_next, r, done, info = env.step(a)

            agent.update_q(s, a, r, s_next)
            agent.update_model(s, a, s_next, r)
            agent.planning()

            ep_reward += r
            hits += int(info['heat_hit'])
            target_distances.append(float(info['target_distance']))
            s = s_next

            if done:
                break

        rewards.append(ep_reward)
        heat_counts.append(hits)
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        if (episode + 1) % 25 == 0:
            avg_r = float(np.mean(rewards[-25:]))
            avg_h = float(np.mean(heat_counts[-25:]))
            avg_d = float(np.mean(target_distances[-25 * cfg.max_steps:])) if target_distances else 0.0
            print(
                f'Episode {episode + 1:4d} | avg_reward={avg_r:7.3f} | '
                f'avg_heat_hits={avg_h:6.2f} | avg_target_dist={avg_d:5.2f} | '
                f'epsilon={epsilon:5.3f}'
            )

        if time.time() - last_log_time >= cfg.log_every_seconds:
            last_log_time = time.time()
            avg_r = float(np.mean(rewards[-25:])) if rewards else 0.0
            avg_h = float(np.mean(heat_counts[-25:])) if heat_counts else 0.0
            print(
                f'Progress | episodes={episode + 1} | elapsed_min={(time.time() - start_time) / 60.0:.1f} | '
                f'avg_reward={avg_r:7.3f} | avg_heat_hits={avg_h:6.2f} | epsilon={epsilon:5.3f}'
            )

    save_policy(agent, output_policy_path)
    print(f'Saved policy: {output_policy_path}')


def main() -> None:
    base = Path(__file__).resolve().parent
    experience_path = base.parent / 'shared' / 'sim_experience.json'
    cfg = Config(
        episodes=_env_int('DYNA_EPISODES', Config.episodes),
        train_seconds=_env_int('DYNA_TRAIN_SECONDS', Config.train_seconds),
        log_every_seconds=_env_int('DYNA_LOG_EVERY_SECONDS', Config.log_every_seconds),
        planning_steps=_env_int('DYNA_PLANNING_STEPS', Config.planning_steps),
        warmup_episodes=_env_int('DYNA_WARMUP_EPISODES', Config.warmup_episodes),
    )
    train(cfg, experience_path)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    main()
