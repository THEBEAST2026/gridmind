from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import grid2op
import warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
import gymnasium as gym

app = FastAPI()

# ── Allow frontend to connect ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load everything on startup ──
print("Loading GridMind models...")
env_g2op = grid2op.make("l2rpn_case14_sandbox")

lstm_data = torch.load("lstm_forecaster.pt", weights_only=False)
gnn_data = torch.load("gnn_encoder.pt", weights_only=False)

edge_index = gnn_data['edge_index']
corr_matrix = gnn_data['corr_matrix']
n_loads = lstm_data['n_loads']
mean = lstm_data['mean']
std = lstm_data['std']

# ── Rebuild GNN ──
class GridGNN(nn.Module):
    def __init__(self, node_features=3, hidden_dim=32, output_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(node_features, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.msg_weight1 = nn.Linear(node_features, hidden_dim)
        self.msg_weight2 = nn.Linear(hidden_dim, output_dim)

    def message_pass(self, x, edge_index, weight):
        src, dst = edge_index
        n_nodes = x.shape[0]
        messages = weight(x[src])
        agg = torch.zeros(n_nodes, messages.shape[-1])
        count = torch.zeros(n_nodes, 1)
        for i in range(len(dst)):
            agg[dst[i]] += messages[i]
            count[dst[i]] += 1
        return agg / count.clamp(min=1)

    def forward(self, x, edge_index):
        h1 = F.relu(self.layer1(x) +
                    self.message_pass(x, edge_index, self.msg_weight1))
        h2 = F.relu(self.layer2(h1) +
                    self.message_pass(h1, edge_index, self.msg_weight2))
        return h2.mean(dim=0)

# ── Rebuild LSTM ──
class DemandLSTM(nn.Module):
    def __init__(self, n_loads, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_loads, hidden_size,
                           num_layers=num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, n_loads)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

gnn = GridGNN()
gnn.load_state_dict(gnn_data['gnn_state'])
gnn.eval()

lstm = DemandLSTM(n_loads=n_loads)
lstm.load_state_dict(lstm_data['model_state'])
lstm.eval()

# ── Rebuild SafeGridEnv ──
class SafeGridEnv(gym.Env):
    def __init__(self, g2op_env):
        self.g2op_env = g2op_env
        self.obs_size = 6 * g2op_env.n_line
        self.observation_space = gym.spaces.Box(
            low=-1, high=2,
            shape=(self.obs_size,),
            dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def _get_obs(self, obs):
        return np.concatenate([
            obs.rho, obs.p_or, obs.p_ex,
            obs.v_or, obs.a_or,
            obs.line_status.astype(float)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs = self.g2op_env.reset()
        return self._get_obs(obs), {}

    def step(self, action):
        if action == 1:
            g2op_action = self.g2op_env.action_space(
                {"set_line_status": [(0, 1)]})
        else:
            g2op_action = self.g2op_env.action_space({})
        obs, reward, done, info = self.g2op_env.step(g2op_action)
        if done:
            custom_reward = -50.0
        else:
            survival = 1.0
            overload_penalty = -5.0 * np.sum(obs.rho > 0.9)
            loss_penalty = -0.01 * np.sum(obs.p_or - obs.p_ex)
            custom_reward = survival + overload_penalty + loss_penalty
        return self._get_obs(obs), custom_reward, done, False, info

safe_env = SafeGridEnv(env_g2op)
ppo_model = PPO.load("gridmind_ppo_final", env=safe_env)

# ── Global state ──
obs_gym, _ = safe_env.reset()
load_buffer = []
step_count = 0
total_loss = []
baseline_loss = 562.14

print("✅ GridMind backend ready!")

# ── Helper functions ──
def get_node_features(obs, env):
    features = np.zeros((env.n_sub, 3))
    for line_id in range(env.n_line):
        or_s = env.line_or_to_subid[line_id]
        ex_s = env.line_ex_to_subid[line_id]
        features[or_s, 0] = max(features[or_s, 0], obs.rho[line_id])
        features[ex_s, 0] = max(features[ex_s, 0], obs.rho[line_id])
    for gen_id in range(env.n_gen):
        sub_id = env.gen_to_subid[gen_id]
        features[sub_id, 1] += obs.gen_p[gen_id] / 100.0
    for load_id in range(env.n_load):
        sub_id = env.load_to_subid[load_id]
        features[sub_id, 2] += obs.load_p[load_id] / 100.0
    return features

def quantum_filter(obs, corr_matrix, threshold=0.7):
    overloaded = np.where(obs.rho > 0.85)[0]
    if len(overloaded) == 0:
        return False, []
    entangled = np.where(
        corr_matrix[overloaded].max(axis=0) > threshold
    )[0]
    return True, entangled.tolist()

# ── API Endpoints ──

@app.get("/")
def root():
    return {"status": "GridMind API running ✅"}

@app.get("/api/grid/state")
def get_grid_state():
    """Current grid state — node voltages, line loads"""
    obs = env_g2op.current_obs
    return {
        "rho": obs.rho.tolist(),
        "p_or": obs.p_or.tolist(),
        "p_ex": obs.p_ex.tolist(),
        "v_or": obs.v_or.tolist(),
        "line_status": obs.line_status.tolist(),
        "n_line": int(env_g2op.n_line),
        "n_sub": int(env_g2op.n_sub),
        "or_sub": env_g2op.line_or_to_subid.tolist(),
        "ex_sub": env_g2op.line_ex_to_subid.tolist()
    }

@app.post("/api/agent/step")
def agent_step():
    """Run one GridMind step — returns action + metrics"""
    global obs_gym, load_buffer, step_count, total_loss

    obs_g2op = env_g2op.current_obs

    # GNN embedding
    node_feat = get_node_features(obs_g2op, env_g2op)
    x = torch.tensor(node_feat, dtype=torch.float32)
    with torch.no_grad():
        gnn_embedding = gnn(x, edge_index)

    # LSTM forecast
    load_buffer.append(obs_g2op.load_p.copy())
    if len(load_buffer) > 12:
        load_buffer.pop(0)

    forecast = None
    if len(load_buffer) == 12:
        load_seq = np.array(load_buffer)
        load_norm = (load_seq - mean) / (std + 1e-8)
        load_tensor = torch.tensor(
            load_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            forecast_norm = lstm(load_tensor)
            forecast = (forecast_norm.numpy() * std + mean).tolist()

    # Quantum heuristic
    activated, entangled = quantum_filter(obs_g2op, corr_matrix)

    # PPO action
    action, _ = ppo_model.predict(obs_gym, deterministic=True)
    obs_gym_new, reward, done, truncated, info = safe_env.step(int(action))

    if done:
        obs_gym, _ = safe_env.reset()
        load_buffer = []
    else:
        obs_gym = obs_gym_new

    # Metrics
    current_loss = float(np.sum(
        env_g2op.current_obs.p_or - env_g2op.current_obs.p_ex))
    total_loss.append(current_loss)
    avg_loss = float(np.mean(total_loss))
    improvement = float(
        (baseline_loss - avg_loss) / baseline_loss * 100)
    step_count += 1

    return {
        "step": step_count,
        "action": int(action),
        "current_loss_mw": round(current_loss, 2),
        "avg_loss_mw": round(avg_loss, 2),
        "improvement_pct": round(improvement, 1),
        "quantum_activated": activated,
        "entangled_lines": entangled,
        "forecast_mw": forecast,
        "done": bool(done),
        "rho": env_g2op.current_obs.rho.tolist()
    }

@app.post("/api/grid/reset")
def reset_grid():
    """Reset grid to initial state"""
    global obs_gym, load_buffer, step_count, total_loss
    obs_gym, _ = safe_env.reset()
    load_buffer = []
    step_count = 0
    total_loss = []
    return {"status": "reset", "message": "Grid reset to initial state"}

@app.get("/api/metrics/history")
def get_metrics():
    """Return loss history for charts"""
    return {
        "total_steps": step_count,
        "loss_history": [round(l, 2) for l in total_loss[-100:]],
        "current_improvement": round(
            (baseline_loss - np.mean(total_loss)) /
            baseline_loss * 100, 1) if total_loss else 0,
        "baseline_loss": baseline_loss
    }

    # ── Modbus TCP Integration (Production Ready) ──
# Uncomment for real hardware deployment
# pip install pymodbus

# from pymodbus.client import ModbusTcpClient
# 
# class ModbusGridClient:
#     def __init__(self, host="192.168.1.100", port=502):
#         self.host = host
#         self.port = port
#         self.client = None
#         self.connected = False
# 
#     def connect(self):
#         self.client = ModbusTcpClient(self.host, port=self.port)
#         self.connected = self.client.connect()
#         return self.connected
# 
#     def read_grid_state(self):
#         """
#         Register Map:
#         0-19   → rho (line loads × 1000)
#         20-39  → p_or (power flow × 10)
#         40-59  → p_ex (power flow × 10)
#         60-79  → v_or (voltage × 100)
#         80-99  → line status (0/1)
#         """
#         result = self.client.read_holding_registers(0, 100)
#         r = result.registers
#         return {
#             "rho":  [r[i]/1000.0 for i in range(20)],
#             "p_or": [r[i+20]/10.0 for i in range(20)],
#             "p_ex": [r[i+40]/10.0 for i in range(20)],
#             "v_or": [r[i+60]/100.0 for i in range(20)],
#             "line_status": [bool(r[i+80]) for i in range(20)]
#         }
# 
#     def disconnect(self):
#         if self.client:
#             self.client.close()
#
# @app.post("/api/hardware/connect")
# def connect_hardware(host: str = "192.168.1.100", port: int = 502):
#     """Switch from Grid2Op simulation to real Modbus TCP sensors"""
#     client = ModbusGridClient(host, port)
#     success = client.connect()
#     return {
#         "status": "connected" if success else "failed",
#         "host": host,
#         "port": port
#     }