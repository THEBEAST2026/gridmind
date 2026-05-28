# GridMind — AI Smart Grid Optimizer

PPO reinforcement learning agent for real-time power grid management.
Reduces transmission loss by **18.2%** vs baseline.

## AI Stack
- **PPO Agent** — switching decisions (stable-baselines3)
- **GNN Encoder** — 14-node topology encoding (PyTorch)
- **LSTM Forecaster** — demand prediction, 99% accuracy
- **Quantum Heuristic** — correlation-based action filtering

## Results
| Metric | Value |
|---|---|
| Loss reduction | 18.2% |
| Steps survived | 200/200 |
| Forecast accuracy | 99% |

## Run locally
pip install -r requirements.txt
python train.py
python gridmind_final.py

## Live Demo
https://THEBEAST2026.github.io/gridmind/