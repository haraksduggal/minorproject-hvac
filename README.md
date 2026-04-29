# HVAC Placement Optimizer — Web Dashboard

A full-stack RL-based HVAC vent placement optimizer with an interactive web frontend.

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Then open **http://localhost:5000** in your browser.

## Features

- **Interactive Floor Map Editor** — Draw walls, floors, and doors with click-and-drag
- **5 Preset Floor Plans** — L-shaped office, multi-room, corridor, open plan, U-shaped
- **3 Visualization Overlays** — Thermal heatmap, airflow direction arrows, clean blueprint
- **RL Optimization** — DQN agent finds minimum HVAC units for ≥95% coverage
- **Live Training Progress** — Watch the agent learn in real time
- **Vent Selection** — Click vents on map or sidebar list to highlight
- **Thermal Analytics** — Coverage %, dead zones, efficiency, convergence chart
- **JSON Export** — Download results with one click
- **Works Offline** — Falls back to in-browser Q-learning if server API is unavailable

## How It Works

1. Draw or select a floor plan (0=wall, 1=floor, 2=door)
2. Adjust cooling radius and training episodes
3. Click "Run Optimization"
4. View results: vent positions, heatmap, and statistics

## Architecture

```
hvac_full/
├── server.py              # Flask backend (serves UI + runs RL API)
├── requirements.txt
├── frontend/
│   └── index.html         # Complete standalone dashboard
└── hvac_rl/
    ├── environment.py      # Thermal model + DQN agent (pure NumPy)
    ├── presets.py           # 5 floor plan presets
    └── visualizer.py        # Terminal renderer (optional CLI use)
```

## Dual Mode

- **With server**: `python server.py` — RL runs on Python backend (faster, full DQN)
- **Without server**: Open `frontend/index.html` directly — RL runs in-browser via JS
