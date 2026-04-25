<!-- HEADER -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,100:020617&height=200&section=header&text=Energy%20Optimising%20HVAC&fontSize=40&fontColor=38bdf8&animation=fadeIn&fontAlignY=35"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2500&pause=800&color=38BDF8&center=true&vCenter=true&width=600&lines=AI+meets+Energy+Efficiency;Smart+HVAC+Placement+using+Reinforcement+Learning;Minimal+Energy.+Maximum+Coverage.;Built+for+real-world+impact"/>
</p>

---

## 🌑 Overview

> *Not your average HVAC system.*

This project uses **Reinforcement Learning (DQN)** to intelligently optimize HVAC vent placement — maximizing cooling efficiency while minimizing energy usage.

No manual tuning. No guesswork.  
Just a system that *learns and adapts*.

---

## 🌐 Live Demo

<p align="center">
  <a href="https://energyoptimisinghvac.netlify.app/">
    <img src="https://img.shields.io/badge/Launch%20App-0f172a?style=for-the-badge&logo=vercel&logoColor=38bdf8"/>
  </a>
</p>

---

## ✨ Features

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,js,html,css,flask"/>
</p>

- 🧩 Interactive floor plan editor  
- 🤖 RL-based vent placement (DQN)  
- 🔥 Dynamic thermal heatmaps  
- 📊 Real-time analytics (coverage, efficiency, dead zones)  
- ⚡ Dual execution (Backend + Browser fallback)  
- 📦 Export results as JSON  

---

## 🧠 How It Works

```text
1. Design floor layout (grid-based)
2. Set cooling radius + episodes
3. RL agent explores placements
4. Learns via reward optimization
5. Outputs best configuration

hvac_full/
├── server.py
├── requirements.txt
├── frontend/
│   └── index.html
└── hvac_rl/
    ├── environment.py
    ├── presets.py
    └── visualizer.py

pip install -r requirements.txt
python server.py
