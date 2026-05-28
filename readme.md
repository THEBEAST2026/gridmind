# GridMind 🔋⚡
### AI-Powered Smart Grid Optimizer — National Buildathon 2026 · Forge India

> Reinforcement learning meets real-time power grid management. GridMind cuts transmission loss by **18.2%** and survives **200/200 steps** — where the random baseline collapses in 8.

---

## 🏆 Results

| Metric | Baseline | GridMind |
|---|---|---|
| Avg Transmission Loss | 562.14 MW | 459.86 MW |
| Steps Survived | ~8 steps | 200/200 ✅ |
| Demand Forecast Accuracy | — | 99% (MAPE < 1%) |
| Quantum Heuristic Active | — | ~34% of steps |
| **Loss Reduction** | — | **18.2%** |

---

## 🧠 AI Stack

```
Grid Observation (120-dim)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│   GNN Encoder   │     │  LSTM Forecaster  │
│  14-node graph  │     │  12-step window   │
│  64-dim embed   │     │  11 load nodes    │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           Quantum Heuristic             │
│   Correlation matrix — entangled lines  │
│   Prunes PPO action space on overload   │
└─────────────────────┬───────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │   PPO Agent    │
             │ 50k timesteps  │
             │ Custom reward  │
             └────────────────┘
                      │
                      ▼
             Switching Decision
```

### Components
- **PPO Agent** — Proximal Policy Optimization (stable-baselines3), trained 50,000 timesteps with custom shaped reward
- **GNN Encoder** — 2-layer Graph Neural Network with message passing over IEEE 14-bus topology, outputs 64-dim embedding
- **LSTM Forecaster** — 2-layer LSTM on 500-step load history, 12-step sliding window, MAPE < 1%
- **Quantum Heuristic** — Correlation matrix identifies statistically "entangled" lines during fault events, prunes action space

---

## 🌐 Live Demo

👉 **[https://thebeast2026.github.io/gridmind/](https://thebeast2026.github.io/gridmind/)**

Features:
- Animated WebGL current-flow background
- Live simulation replay (200 steps)
- Interactive quantum fault simulator
- 14-node grid topology visualization
- Real-time line load charts

---

## 🗂 Project Structure

```
gridmind/
├── index.html              # Live demo frontend
├── backend.py              # FastAPI backend (optional live deployment)
├── train.py                # PPO training script
├── gridmind_final.py       # Full GridMind evaluation pipeline
├── gnn.py                  # GNN encoder + quantum heuristic
├── lstm.py                 # LSTM demand forecaster
├── evaluate.py             # Evaluation vs baseline
├── evaluatev2.py           # Episode-by-episode evaluation
├── baseline.py             # Random action baseline
├── gnn_encoder.pt          # Trained GNN weights
├── lstm_forecaster.pt      # Trained LSTM weights
├── gridmind_ppo_final.zip  # Trained PPO agent
└── requirements.txt
```

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the PPO agent (optional — pretrained weights included)
python train.py

# 3. Train GNN + build quantum heuristic
python gnn.py

# 4. Train LSTM forecaster
python lstm.py

# 5. Run full evaluation
python gridmind_final.py

# 6. Run backend API (optional)
uvicorn backend:app --reload
```

---

## 🔧 Environment

- **Grid simulator** — Grid2Op `l2rpn_case14_sandbox` (IEEE 14-bus)
- **Observation** — 120-dim vector: ρ, P_or, P_ex, V_or, A_or, line_status
- **Actions** — 3 discrete: do-nothing, reconnect line, do-nothing
- **Reward** — +1 survival · −5 per overloaded line · −0.01×loss · −50 game-over

---

## 📦 Requirements

```
grid2op
stable-baselines3
torch
gymnasium
numpy
fastapi
uvicorn
```

---

*Built for National Buildathon 2026 · Forge India · AI Track*
