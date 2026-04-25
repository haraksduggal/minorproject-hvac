"""
Flask server: serves the frontend and provides API endpoints
for running the RL optimizer and returning results as JSON.
"""

import json
import os
import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hvac_rl.environment import ThermalConfig, train_hvac_placement, compute_cooling_map, coverage_ratio

app = Flask(__name__, static_folder="frontend", static_url_path="")


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    grid = np.array(data["grid"], dtype=int)
    radius = int(data.get("radius", 5))
    episodes = int(data.get("episodes", 400))

    config = ThermalConfig(cooling_radius=radius)
    result = train_hvac_placement(grid, config, n_episodes=episodes, verbose=False)

    positions = [tuple(int(x) for x in p) for p in result["best_positions"]]
    cooling = compute_cooling_map(grid, positions, config)
    cov = coverage_ratio(grid, cooling, config)

    # Compute per-vent stats
    vent_stats = []
    for i, pos in enumerate(positions):
        solo = compute_cooling_map(grid, [pos], config)
        solo_cov = coverage_ratio(grid, solo, config)
        vent_stats.append({
            "id": i + 1,
            "row": int(pos[0]),
            "col": int(pos[1]),
            "solo_coverage": round(solo_cov * 100, 1),
        })

    # Compute dead zones
    walkable = grid > 0
    dead_zones = int(np.sum((cooling < config.min_coverage_threshold) & walkable))

    # Average cooling
    walk_cells = np.sum(walkable)
    avg_cooling = float(np.sum(cooling[walkable]) / walk_cells) if walk_cells > 0 else 0

    return jsonify({
        "positions": [{"row": int(p[0]), "col": int(p[1])} for p in positions],
        "n_units": result["best_n_units"],
        "coverage": round(result["best_coverage"], 1),
        "heatmap": cooling.tolist(),
        "vent_stats": vent_stats,
        "dead_zones": dead_zones,
        "avg_cooling": round(avg_cooling * 100, 1),
        "walkable_cells": int(walk_cells),
        "efficiency": round(float(walk_cells) / max(1, result["best_n_units"]), 1),
        "history": [
            {"ep": h["episode"], "cov": h["best_coverage"]}
            for h in result["training_history"][::max(1, len(result["training_history"])//80)]
        ],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  HVAC Optimizer — Web Dashboard               ║")
    print(f"  ║  Open http://localhost:{port} in your browser    ║")
    print(f"  ╚═══════════════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=port, debug=False)
