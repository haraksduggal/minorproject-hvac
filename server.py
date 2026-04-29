"""
Flask server v2: furniture support, tonnage estimation, max AC limit.
"""
import json, os, sys, math
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hvac_rl.environment import (
    ThermalConfig, train_hvac_placement, compute_cooling_map,
    coverage_ratio, estimate_tonnage, TONNAGE_OPTIONS,
)

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
    max_ac = data.get("max_ac_limit")
    if max_ac is not None:
        max_ac = int(max_ac)

    config = ThermalConfig(cooling_radius=radius)
    result = train_hvac_placement(grid, config, n_episodes=episodes, verbose=False, max_ac_limit=max_ac)

    positions = [tuple(int(x) for x in p) for p in result["best_positions"]]
    cooling = compute_cooling_map(grid, positions, config)
    cov = coverage_ratio(grid, cooling, config)

    # Tonnage estimation per vent
    vent_info = []
    total_tons = 0.0
    for i, pos in enumerate(positions):
        tons = estimate_tonnage(grid, pos, config, positions)
        total_tons += tons
        vent_info.append({
            "id": i + 1,
            "row": int(pos[0]),
            "col": int(pos[1]),
            "tonnage": tons,
        })

    walkable = grid > 0
    dead_zones = int(np.sum((cooling < config.min_coverage_threshold) & walkable))
    walk_cells = int(np.sum(walkable))
    avg_cool = float(np.sum(cooling[walkable]) / walk_cells) if walk_cells > 0 else 0
    furn_cells = int(np.sum((grid == 3) | (grid == 4)))

    return jsonify({
        "positions": [{"row": v["row"], "col": v["col"], "tonnage": v["tonnage"]} for v in vent_info],
        "n_units": result["best_n_units"],
        "coverage": round(result["best_coverage"], 1),
        "heatmap": cooling.tolist(),
        "vent_stats": vent_info,
        "total_tonnage": total_tons,
        "dead_zones": dead_zones,
        "avg_cooling": round(avg_cool * 100, 1),
        "walkable_cells": walk_cells,
        "furniture_cells": furn_cells,
        "efficiency": round(walk_cells / max(1, result["best_n_units"]), 1),
        "max_ac_limit": result["max_ac_limit"],
        "tonnage_options": TONNAGE_OPTIONS,
        "history": [
            {"ep": h["episode"], "cov": h["best_coverage"]}
            for h in result["training_history"][::max(1, len(result["training_history"])//80)]
        ],
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  HVAC Optimizer v2 — Web Dashboard            ║")
    print(f"  ║  Open http://localhost:{port} in your browser    ║")
    print(f"  ╚═══════════════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=port, debug=False)
