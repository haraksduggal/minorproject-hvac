"""
HVAC Placement Optimization using Reinforcement Learning
=========================================================
Environment: Grid-based floor map where the agent places HVAC units.
State: Current floor coverage map + placed units.
Action: Place an HVAC unit at a grid cell.
Reward: Coverage improvement minus cost penalty per unit.

The goal is to find the MINIMUM number of HVAC units and their
OPTIMAL placement to achieve full cooling coverage of walkable space.
"""

import numpy as np
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random


# ─── Thermal Model ────────────────────────────────────────────────────────────

@dataclass
class ThermalConfig:
    """Physics-inspired thermal parameters for HVAC cooling simulation."""
    cooling_radius: int = 5
    cooling_power: float = 1.0
    decay_rate: float = 0.15
    wall_attenuation: float = 0.85
    min_coverage_threshold: float = 0.3
    target_coverage_pct: float = 0.95


def compute_cooling_map(
    floor_map: np.ndarray,
    hvac_positions: List[Tuple[int, int]],
    config: ThermalConfig
) -> np.ndarray:
    """
    Compute aggregate cooling intensity at every walkable cell.
    Uses exponential decay with wall attenuation via raycasting.
    """
    rows, cols = floor_map.shape
    cooling = np.zeros_like(floor_map, dtype=np.float64)

    for (hr, hc) in hvac_positions:
        r_min = max(0, hr - config.cooling_radius)
        r_max = min(rows, hr + config.cooling_radius + 1)
        c_min = max(0, hc - config.cooling_radius)
        c_max = min(cols, hc + config.cooling_radius + 1)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if floor_map[r, c] == 0:
                    continue
                dist = math.sqrt((r - hr) ** 2 + (c - hc) ** 2)
                if dist > config.cooling_radius:
                    continue
                walls_crossed = _raycast_walls(floor_map, hr, hc, r, c)
                attenuation = config.wall_attenuation ** walls_crossed
                intensity = config.cooling_power * math.exp(-config.decay_rate * dist) * attenuation
                cooling[r, c] += intensity

    return cooling


def _raycast_walls(floor_map: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    """Bresenham-style raycast counting wall cells crossed."""
    walls = 0
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0

    while True:
        if (r, c) != (r0, c0) and (r, c) != (r1, c1):
            if floor_map[r, c] == 0:
                walls += 1
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return walls


def coverage_ratio(floor_map, cooling_map, config):
    walkable = floor_map > 0
    n_walkable = np.sum(walkable)
    if n_walkable == 0:
        return 1.0
    covered = np.sum((cooling_map >= config.min_coverage_threshold) & walkable)
    return float(covered / n_walkable)


# ─── RL Environment ───────────────────────────────────────────────────────────

class HVACPlacementEnv:
    """
    RL Environment for HVAC placement.

    Episode flow: Agent places units one at a time until coverage target
    is met or max units exhausted.

    Reward shaping:
        +100 * coverage_improvement
        -2.0 per unit placed
        +50 bonus when target reached
        -1.0 for duplicate placement
    """

    def __init__(self, floor_map: np.ndarray, config: Optional[ThermalConfig] = None):
        self.floor_map = floor_map.copy()
        self.config = config or ThermalConfig()
        self.rows, self.cols = floor_map.shape

        self.valid_cells = list(zip(*np.where(floor_map > 0)))
        self.n_valid = len(self.valid_cells)
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.valid_cells)}

        self.max_units = max(3, self.n_valid // 8)
        self.reset()

    def reset(self):
        self.placed_units: List[Tuple[int, int]] = []
        self.cooling_map = np.zeros_like(self.floor_map, dtype=np.float64)
        self.current_coverage = 0.0
        self.done = False
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        cell_cooling = np.array([self.cooling_map[r, c] for r, c in self.valid_cells], dtype=np.float32)
        placement_mask = np.zeros(self.n_valid, dtype=np.float32)
        for pos in self.placed_units:
            if pos in self.cell_to_idx:
                placement_mask[self.cell_to_idx[pos]] = 1.0
        return np.concatenate([cell_cooling, placement_mask, [self.current_coverage]])

    @property
    def state_dim(self):
        return 2 * self.n_valid + 1

    @property
    def action_dim(self):
        return self.n_valid

    def step(self, action_idx: int):
        if self.done:
            return self._get_state(), 0.0, True, {}

        pos = self.valid_cells[action_idx]
        reward = 0.0

        if pos in self.placed_units:
            reward -= 1.0
            self.done = True
            return self._get_state(), reward, True, {"reason": "duplicate"}

        self.placed_units.append(pos)
        self.steps += 1

        self.cooling_map = compute_cooling_map(self.floor_map, self.placed_units, self.config)
        new_coverage = coverage_ratio(self.floor_map, self.cooling_map, self.config)

        coverage_gain = new_coverage - self.current_coverage
        self.current_coverage = new_coverage

        reward += 100.0 * coverage_gain
        reward -= 2.0

        if self.current_coverage >= self.config.target_coverage_pct:
            reward += 50.0
            self.done = True

        if self.steps >= self.max_units:
            self.done = True

        info = {
            "coverage": self.current_coverage,
            "n_units": len(self.placed_units),
            "positions": list(self.placed_units),
        }

        return self._get_state(), reward, self.done, info


# ─── DQN Agent (Pure NumPy) ──────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states), np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """Fully-connected Q-network in pure NumPy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 64)):
        self.layers = []
        dims = [state_dim] + list(hidden_sizes) + [action_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * scale
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self.layers.append((W, b))

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
        return x[0] if single else x

    def copy_from(self, other: 'QNetwork'):
        for i in range(len(self.layers)):
            self.layers[i] = (other.layers[i][0].copy(), other.layers[i][1].copy())


class DQNAgent:
    """
    DQN with experience replay, target network, epsilon-greedy,
    invalid action masking, and Adam optimizer.
    """

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, tau=0.01, buffer_size=10000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.copy_from(self.q_net)

        self.replay = ReplayBuffer(buffer_size)
        self.adam_m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.q_net.layers]
        self.adam_v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.q_net.layers]
        self.adam_t = 0

    def select_action(self, state, placed_mask):
        if random.random() < self.epsilon:
            valid = np.where(placed_mask == 0)[0]
            if len(valid) == 0:
                return random.randint(0, self.action_dim - 1)
            return int(np.random.choice(valid))

        q_vals = self.q_net.forward(state)
        q_vals[placed_mask == 1] = -np.inf
        return int(np.argmax(q_vals))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        q_all = self.q_net.forward(states)
        q_current = q_all[np.arange(len(actions)), actions]
        q_next = self.target_net.forward(next_states)
        q_target = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)
        td_error = q_current - q_target
        loss = np.mean(td_error ** 2)

        self._backward(states, actions, td_error)

        for i in range(len(self.q_net.layers)):
            qW, qb = self.q_net.layers[i]
            tW, tb = self.target_net.layers[i]
            self.target_net.layers[i] = (
                self.tau * qW + (1 - self.tau) * tW,
                self.tau * qb + (1 - self.tau) * tb,
            )
        return float(loss)

    def _backward(self, states, actions, td_error):
        batch_size = len(states)
        n_layers = len(self.q_net.layers)

        activations = [states.astype(np.float32)]
        x = states.astype(np.float32)
        for i, (W, b) in enumerate(self.q_net.layers):
            x = x @ W + b
            if i < n_layers - 1:
                x = np.maximum(0, x)
            activations.append(x)

        grad_output = np.zeros_like(activations[-1])
        grad_output[np.arange(batch_size), actions] = 2.0 * td_error / batch_size

        self.adam_t += 1
        grad = grad_output

        for i in reversed(range(n_layers)):
            W, b = self.q_net.layers[i]
            inp = activations[i]
            dW = np.clip(inp.T @ grad, -1.0, 1.0)
            db = np.clip(np.sum(grad, axis=0), -1.0, 1.0)

            mW, mb = self.adam_m[i]
            vW, vb = self.adam_v[i]
            mW = 0.9 * mW + 0.1 * dW
            mb = 0.9 * mb + 0.1 * db
            vW = 0.999 * vW + 0.001 * (dW ** 2)
            vb = 0.999 * vb + 0.001 * (db ** 2)
            self.adam_m[i] = (mW, mb)
            self.adam_v[i] = (vW, vb)

            mW_hat = mW / (1 - 0.9 ** self.adam_t)
            mb_hat = mb / (1 - 0.9 ** self.adam_t)
            vW_hat = vW / (1 - 0.999 ** self.adam_t)
            vb_hat = vb / (1 - 0.999 ** self.adam_t)

            W_new = W - self.lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b_new = b - self.lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
            self.q_net.layers[i] = (W_new, b_new)

            if i > 0:
                grad = grad @ W.T
                grad = grad * (activations[i] > 0).astype(np.float32)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train_hvac_placement(floor_map, config=None, n_episodes=500, verbose=True):
    config = config or ThermalConfig()
    env = HVACPlacementEnv(floor_map, config)

    agent = DQNAgent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        lr=0.001, gamma=0.95, epsilon_start=1.0,
        epsilon_end=0.05, epsilon_decay=0.993,
        batch_size=32, buffer_size=5000,
    )

    best_result = {"positions": [], "n_units": env.max_units + 1, "coverage": 0.0}
    history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        placed_mask = np.zeros(env.action_dim, dtype=np.float32)

        while not env.done:
            action = agent.select_action(state, placed_mask)
            next_state, reward, done, info = env.step(action)
            agent.replay.push(state, action, reward, next_state, done)
            agent.train_step()
            placed_mask[action] = 1.0
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        ep_coverage = info.get("coverage", 0.0)
        ep_units = info.get("n_units", 0)
        ep_positions = info.get("positions", [])

        is_better = False
        if ep_coverage >= config.target_coverage_pct:
            if ep_units < best_result["n_units"]:
                is_better = True
            elif ep_units == best_result["n_units"] and ep_coverage > best_result["coverage"]:
                is_better = True
        elif ep_coverage > best_result["coverage"] and best_result["coverage"] < config.target_coverage_pct:
            is_better = True

        if is_better:
            best_result = {"positions": ep_positions, "n_units": ep_units, "coverage": ep_coverage}

        history.append({
            "episode": ep + 1, "reward": round(total_reward, 2),
            "coverage": round(ep_coverage * 100, 1), "n_units": ep_units,
            "epsilon": round(agent.epsilon, 4),
            "best_units": best_result["n_units"],
            "best_coverage": round(best_result["coverage"] * 100, 1),
        })

        if verbose and (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1:>4}/{n_episodes}  |  "
                  f"Best: {best_result['n_units']} units @ {best_result['coverage']*100:.1f}% cov  |  "
                  f"ε={agent.epsilon:.3f}")

    return {
        "best_positions": best_result["positions"],
        "best_n_units": best_result["n_units"],
        "best_coverage": round(best_result["coverage"] * 100, 2),
        "training_history": history,
        "floor_shape": list(floor_map.shape),
        "valid_cells": len(env.valid_cells),
    }
