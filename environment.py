"""
HVAC Placement Optimization using Reinforcement Learning
=========================================================
v2: Adds AC tonnage estimation, furniture cells, and max-AC cap.

Cell types:
  0 = wall
  1 = walkable floor
  2 = door/opening
  3 = table (blocks placement, partially blocks airflow)
  4 = chair (blocks placement, partially blocks airflow)

Tonnage model:
  Each placed AC gets a tonnage rating (0.75, 1.0, 1.5, 2.0 tons)
  based on Voronoi-zone area and furniture density.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

TONNAGE_OPTIONS = [0.75, 1.0, 1.5, 2.0]


@dataclass
class ThermalConfig:
    cooling_radius: int = 5
    cooling_power: float = 1.0
    decay_rate: float = 0.15
    wall_attenuation: float = 0.85
    furniture_attenuation: float = 0.95
    min_coverage_threshold: float = 0.3
    target_coverage_pct: float = 0.95
    cell_area_sqft: float = 25.0
    btu_per_sqft: float = 25.0
    furniture_heat_factor: float = 1.15


def compute_cooling_map(floor_map, hvac_positions, config):
    rows, cols = floor_map.shape
    cooling = np.zeros_like(floor_map, dtype=np.float64)
    for (hr, hc) in hvac_positions:
        for r in range(max(0, hr - config.cooling_radius), min(rows, hr + config.cooling_radius + 1)):
            for c in range(max(0, hc - config.cooling_radius), min(cols, hc + config.cooling_radius + 1)):
                if floor_map[r, c] == 0:
                    continue
                dist = math.hypot(r - hr, c - hc)
                if dist > config.cooling_radius:
                    continue
                walls, furn = _raycast_obstacles(floor_map, hr, hc, r, c)
                atten = (config.wall_attenuation ** walls) * (config.furniture_attenuation ** furn)
                cooling[r, c] += config.cooling_power * math.exp(-config.decay_rate * dist) * atten
    return cooling


def _raycast_obstacles(floor_map, r0, c0, r1, c1):
    walls = furn = 0
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        if (r, c) != (r0, c0) and (r, c) != (r1, c1):
            v = floor_map[r, c]
            if v == 0: walls += 1
            elif v in (3, 4): furn += 1
        if r == r1 and c == c1: break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 < dr: err += dr; c += sc
    return walls, furn


def coverage_ratio(floor_map, cooling_map, config):
    walkable = floor_map > 0
    n = np.sum(walkable)
    if n == 0: return 1.0
    return float(np.sum((cooling_map >= config.min_coverage_threshold) & walkable) / n)


def estimate_tonnage(floor_map, position, config, all_positions):
    rows, cols = floor_map.shape
    pr, pc = position
    zone_cells = zone_furn = 0
    for r in range(rows):
        for c in range(cols):
            if floor_map[r, c] == 0: continue
            nearest = min(all_positions, key=lambda p: math.hypot(r - p[0], c - p[1]))
            if nearest == (pr, pc):
                zone_cells += 1
                if floor_map[r, c] in (3, 4): zone_furn += 1
    if zone_cells == 0: return TONNAGE_OPTIONS[0]
    area = zone_cells * config.cell_area_sqft
    load = 1.0 + (config.furniture_heat_factor - 1.0) * (zone_furn / zone_cells)
    tons = area * config.btu_per_sqft * load / 12000.0
    for t in TONNAGE_OPTIONS:
        if t >= tons: return t
    return TONNAGE_OPTIONS[-1]


class HVACPlacementEnv:
    def __init__(self, floor_map, config=None, max_ac_limit=None):
        self.floor_map = floor_map.copy()
        self.config = config or ThermalConfig()
        self.rows, self.cols = floor_map.shape
        self.valid_cells = [(r, c) for r, c in zip(*np.where(floor_map > 0)) if floor_map[r, c] in (1, 2)]
        self.n_valid = len(self.valid_cells)
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.valid_cells)}
        auto_max = max(3, self.n_valid // 8)
        self.max_units = min(max_ac_limit, auto_max) if max_ac_limit else auto_max
        self.reset()

    def reset(self):
        self.placed_units = []
        self.cooling_map = np.zeros_like(self.floor_map, dtype=np.float64)
        self.current_coverage = 0.0
        self.done = False
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        cc = np.array([self.cooling_map[r, c] for r, c in self.valid_cells], dtype=np.float32)
        pm = np.zeros(self.n_valid, dtype=np.float32)
        for pos in self.placed_units:
            if pos in self.cell_to_idx: pm[self.cell_to_idx[pos]] = 1.0
        return np.concatenate([cc, pm, [self.current_coverage]])

    @property
    def state_dim(self): return 2 * self.n_valid + 1
    @property
    def action_dim(self): return self.n_valid

    def step(self, action_idx):
        if self.done: return self._get_state(), 0.0, True, {}
        pos = self.valid_cells[action_idx]
        if pos in self.placed_units:
            self.done = True
            return self._get_state(), -1.0, True, {"reason": "duplicate"}
        self.placed_units.append(pos)
        self.steps += 1
        self.cooling_map = compute_cooling_map(self.floor_map, self.placed_units, self.config)
        new_cov = coverage_ratio(self.floor_map, self.cooling_map, self.config)
        gain = new_cov - self.current_coverage
        self.current_coverage = new_cov
        reward = 100.0 * gain - 2.0
        if self.current_coverage >= self.config.target_coverage_pct:
            reward += 50.0; self.done = True
        if self.steps >= self.max_units: self.done = True
        return self._get_state(), reward, self.done, {
            "coverage": self.current_coverage,
            "n_units": len(self.placed_units),
            "positions": list(self.placed_units),
        }


class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buf, self.cap, self.pos = [], cap, 0
    def push(self, *args):
        if len(self.buf) < self.cap: self.buf.append(None)
        self.buf[self.pos] = args; self.pos = (self.pos + 1) % self.cap
    def sample(self, n):
        b = random.sample(self.buf, min(n, len(self.buf)))
        return tuple(np.array(x) for x in zip(*b))
    def __len__(self): return len(self.buf)


class QNetwork:
    def __init__(self, sd, ad, hs=(128, 64)):
        self.layers = []
        dims = [sd] + list(hs) + [ad]
        for i in range(len(dims)-1):
            W = np.random.randn(dims[i], dims[i+1]).astype(np.float32) * np.sqrt(2.0/dims[i])
            self.layers.append((W, np.zeros(dims[i+1], dtype=np.float32)))
    def forward(self, x):
        s = x.ndim == 1
        if s: x = x[np.newaxis, :]
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers)-1: x = np.maximum(0, x)
        return x[0] if s else x
    def copy_from(self, o):
        for i in range(len(self.layers)):
            self.layers[i] = (o.layers[i][0].copy(), o.layers[i][1].copy())


class DQNAgent:
    def __init__(self, sd, ad, lr=.001, gamma=.99, es=1., ee=.05, ed=.995, bs=64, tau=.01, bufsz=10000):
        self.ad, self.gamma, self.lr, self.epsilon = ad, gamma, lr, es
        self.ee, self.ed, self.bs, self.tau = ee, ed, bs, tau
        self.q = QNetwork(sd, ad); self.tgt = QNetwork(sd, ad); self.tgt.copy_from(self.q)
        self.replay = ReplayBuffer(bufsz)
        self.am = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.q.layers]
        self.av = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.q.layers]
        self.at = 0

    def select_action(self, state, mask):
        if random.random() < self.epsilon:
            v = np.where(mask == 0)[0]
            return int(np.random.choice(v)) if len(v) else random.randint(0, self.ad-1)
        qv = self.q.forward(state); qv[mask == 1] = -np.inf
        return int(np.argmax(qv))

    def train_step(self):
        if len(self.replay) < self.bs: return 0.
        S, A, R, S2, D = self.replay.sample(self.bs)
        A = A.astype(int)
        R = R.astype(np.float32)
        D = D.astype(np.float32)
        qa = self.q.forward(S); qc = qa[np.arange(len(A)), A]
        qt = R + self.gamma * np.max(self.tgt.forward(S2), axis=1) * (1 - D)
        td = qc - qt; self._bk(S, A, td)
        for i in range(len(self.q.layers)):
            qW, qb = self.q.layers[i]; tW, tb = self.tgt.layers[i]
            self.tgt.layers[i] = (self.tau*qW+(1-self.tau)*tW, self.tau*qb+(1-self.tau)*tb)
        return float(np.mean(td**2))

    def _bk(self, S, A, td):
        bs = len(S); nl = len(self.q.layers)
        acts = [S.astype(np.float32)]; x = S.astype(np.float32)
        for i, (W, b) in enumerate(self.q.layers):
            x = x @ W + b
            if i < nl-1: x = np.maximum(0, x)
            acts.append(x)
        go = np.zeros_like(acts[-1]); go[np.arange(bs), A] = 2.*td/bs
        self.at += 1; g = go
        for i in reversed(range(nl)):
            W, b = self.q.layers[i]; inp = acts[i]
            dW = np.clip(inp.T @ g, -1, 1); db = np.clip(np.sum(g, 0), -1, 1)
            mW, mb = self.am[i]; vW, vb = self.av[i]
            mW = .9*mW+.1*dW; mb = .9*mb+.1*db
            vW = .999*vW+.001*(dW**2); vb = .999*vb+.001*(db**2)
            self.am[i] = (mW, mb); self.av[i] = (vW, vb)
            t = self.at
            self.q.layers[i] = (
                W - self.lr*(mW/(1-.9**t))/(np.sqrt(vW/(1-.999**t))+1e-8),
                b - self.lr*(mb/(1-.9**t))/(np.sqrt(vb/(1-.999**t))+1e-8))
            if i > 0: g = (g @ W.T) * (acts[i] > 0).astype(np.float32)

    def decay_epsilon(self):
        self.epsilon = max(self.ee, self.epsilon * self.ed)


def train_hvac_placement(floor_map, config=None, n_episodes=500, verbose=True, max_ac_limit=None):
    config = config or ThermalConfig()
    env = HVACPlacementEnv(floor_map, config, max_ac_limit=max_ac_limit)
    agent = DQNAgent(env.state_dim, env.action_dim, lr=.001, gamma=.95, es=1., ee=.05, ed=.993, bs=32, bufsz=5000)
    best = {"positions": [], "n_units": env.max_units + 1, "coverage": 0.0}
    history = []
    for ep in range(n_episodes):
        state = env.reset(); tr = 0.; pm = np.zeros(env.action_dim, dtype=np.float32)
        while not env.done:
            a = agent.select_action(state, pm)
            ns, r, d, info = env.step(a)
            agent.replay.push(state, a, r, ns, float(d))
            agent.train_step(); pm[a] = 1.; state = ns; tr += r
        agent.decay_epsilon()
        ec, eu, ep2 = info.get("coverage", 0.), info.get("n_units", 0), info.get("positions", [])
        ib = False
        if ec >= config.target_coverage_pct:
            if eu < best["n_units"]: ib = True
            elif eu == best["n_units"] and ec > best["coverage"]: ib = True
        elif ec > best["coverage"] and best["coverage"] < config.target_coverage_pct: ib = True
        if ib: best = {"positions": ep2, "n_units": eu, "coverage": ec}
        history.append({"episode": ep+1, "best_units": best["n_units"], "best_coverage": round(best["coverage"]*100, 1)})
        if verbose and (ep+1) % 50 == 0:
            print(f"  Episode {ep+1:>4}/{n_episodes}  |  Best: {best['n_units']} units @ {best['coverage']*100:.1f}%  |  ε={agent.epsilon:.3f}")
    return {
        "best_positions": best["positions"],
        "best_n_units": best["n_units"],
        "best_coverage": round(best["coverage"] * 100, 2),
        "training_history": history,
        "floor_shape": list(floor_map.shape),
        "valid_cells": len(env.valid_cells),
        "max_ac_limit": env.max_units,
    }
