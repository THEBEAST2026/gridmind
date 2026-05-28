import grid2op
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# ── Load environment ──
env = grid2op.make("l2rpn_case14_sandbox")

# ── Step 1: Extract continuous load history ──
print("Extracting load history...")

# Force same chronic every time (no jumps in data)
env.chronics_handler.tell_id(0)
obs = env.reset()

all_loads = []
for step in range(500):
    action = env.action_space({})
    obs, reward, done, info = env.step(action)
    all_loads.append(obs.load_p.copy())
    if done:
        break  # stop cleanly, no reset jumps

load_history = np.array(all_loads)
print(f"Load history shape: {load_history.shape}")

# ── Step 2: Normalize (Z-score) ──
mean = load_history.mean(axis=0)
std = load_history.std(axis=0) + 1e-8  # avoid divide by zero
load_normalized = (load_history - mean) / std
print(f"Normalized — mean≈0: {load_normalized.mean():.4f}")

# ── Step 3: Create sequences ──
def create_sequences(data, seq_length=12):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(load_normalized, seq_length=12)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()
print(f"Input shape  (Batch, Seq, Features): {X_tensor.shape}")
print(f"Output shape (Batch, Features):      {y_tensor.shape}")

# ── Step 4: Build LSTM ──
class DemandLSTM(nn.Module):
    def __init__(self, n_loads, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_loads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, n_loads)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep only

n_loads = load_history.shape[1]
model = DemandLSTM(n_loads=n_loads)
print(f"\nLSTM model built — predicting {n_loads} loads")
print(model)

# ── Step 5: Train ──
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("\nTraining LSTM...")
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/50 — Loss: {avg_loss:.6f}")

# ── Step 6: Evaluate ──
model.eval()
with torch.no_grad():
    sample_input = X_tensor[:10]
    predictions = model(sample_input)
    
    # Denormalize for real MW values
    pred_mw = predictions.numpy() * std + mean
    actual_mw = y_tensor[:10].numpy() * std + mean
    
    mape = np.mean(np.abs(pred_mw - actual_mw) / (actual_mw + 1e-8)) * 100
    print(f"\nMAPE (Mean Absolute % Error): {mape:.2f}%")
    print(f"Target: below 10%")
    
    if mape < 10:
        print("✅ LSTM forecaster is accurate enough for GridMind!")
    else:
        print("⚠️ Need more training — increase epochs to 100")

# ── Step 7: Save ──
torch.save({
    'model_state': model.state_dict(),
    'mean': mean,
    'std': std,
    'n_loads': n_loads
}, 'lstm_forecaster.pt')

print("\nLSTM saved to lstm_forecaster.pt")
print("Done! ✅")