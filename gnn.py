import grid2op
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# ── Load environment ──
env = grid2op.make("l2rpn_case14_sandbox")
obs = env.reset()

# ── Step 1: Build edge index ──
or_sub = env.line_or_to_subid
ex_sub = env.line_ex_to_subid

# Bidirectional edges (both directions for each line)
edge_index = torch.tensor([
    list(or_sub) + list(ex_sub),
    list(ex_sub) + list(or_sub)
], dtype=torch.long)

print(f"Edge index shape: {edge_index.shape}")
print("Graph structure ready!")

# ── Step 2: Build GNN (without PyG dependency) ──
class GridGNN(nn.Module):
    def __init__(self, node_features=3, hidden_dim=32, output_dim=64):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Layer 1: node features → hidden
        self.layer1 = nn.Linear(node_features, hidden_dim)
        
        # Layer 2: hidden → output
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
        # Message passing weights
        self.msg_weight1 = nn.Linear(node_features, hidden_dim)
        self.msg_weight2 = nn.Linear(hidden_dim, output_dim)
        
    def message_pass(self, x, edge_index, weight):
        """Aggregate neighbor features for each node"""
        src, dst = edge_index
        n_nodes = x.shape[0]
        
        # Transform source node features
        messages = weight(x[src])
        
        # Aggregate messages at destination nodes (mean aggregation)
        agg = torch.zeros(n_nodes, messages.shape[-1])
        count = torch.zeros(n_nodes, 1)
        
        for i in range(len(dst)):
            agg[dst[i]] += messages[i]
            count[dst[i]] += 1
        
        # Avoid division by zero
        count = count.clamp(min=1)
        agg = agg / count
        
        return agg
    
    def forward(self, x, edge_index):
        # Layer 1: combine self + neighbor features
        self_feat1 = self.layer1(x)
        neigh_feat1 = self.message_pass(x, edge_index, self.msg_weight1)
        h1 = F.relu(self_feat1 + neigh_feat1)
        
        # Layer 2
        self_feat2 = self.layer2(h1)
        neigh_feat2 = self.message_pass(h1, edge_index, self.msg_weight2)
        h2 = F.relu(self_feat2 + neigh_feat2)
        
        # Global mean pooling → single embedding for entire graph
        graph_embedding = h2.mean(dim=0)
        
        return graph_embedding

# ── Step 3: Build node features from observation ──
def get_node_features(obs, env):
    """Extract 3 features per substation node"""
    n_sub = env.n_sub
    features = np.zeros((n_sub, 3))
    
    # Feature 1: max line load ratio (rho) at each substation
    for line_id in range(env.n_line):
        or_s = env.line_or_to_subid[line_id]
        ex_s = env.line_ex_to_subid[line_id]
        features[or_s, 0] = max(features[or_s, 0], obs.rho[line_id])
        features[ex_s, 0] = max(features[ex_s, 0], obs.rho[line_id])
    
    # Feature 2: total generation at each substation
    for gen_id in range(env.n_gen):
        sub_id = env.gen_to_subid[gen_id]
        features[sub_id, 1] += obs.gen_p[gen_id] /100.0  # normalize by max gen capacity (100 MW)
    
    # Feature 3: total load at each substation
    for load_id in range(env.n_load):
        sub_id = env.load_to_subid[load_id]
        features[sub_id, 2] += obs.load_p[load_id] /100.0  # normalize by max load capacity (100 MW)
    
    return features

# ── Step 4: Test GNN forward pass ──
print("\nTesting GNN forward pass...")
gnn = GridGNN(node_features=3, hidden_dim=32, output_dim=64)

node_features = get_node_features(obs, env)
x = torch.tensor(node_features, dtype=torch.float32)

embedding = gnn(x, edge_index)
print(f"Node features shape: {x.shape}")
print(f"Graph embedding shape: {embedding.shape}")
print(f"Embedding sample: {embedding[:5].detach().numpy()}")

# ── Step 5: Quantum heuristic ──
print("\nBuilding quantum entanglement heuristic...")

# Collect load history for correlation matrix
env.chronics_handler.tell_id(0)
obs = env.reset()
load_history = []

for _ in range(200):
    action = env.action_space({})
    obs, reward, done, info = env.step(action)
    load_history.append(obs.rho.copy())
    if done:
        break

load_history = np.array(load_history)

# Correlation matrix between lines
corr_matrix = np.corrcoef(load_history.T)
print(f"Correlation matrix shape: {corr_matrix.shape}")

def quantum_filter_actions(obs, all_actions, corr_matrix, threshold=0.7):
    """
    Quantum-inspired action filter:
    Find lines 'entangled' with overloaded lines
    and prioritize actions involving those lines
    """
    overloaded = np.where(obs.rho > 0.85)[0]
    
    if len(overloaded) == 0:
        return list(range(min(10, env.n_line)))  # no overload, limit actions
    
    # Find correlated (entangled) lines
    entangled = np.where(
        corr_matrix[overloaded].max(axis=0) > threshold
    )[0]
    
    print(f"  Overloaded lines: {overloaded}")
    print(f"  Quantum-entangled lines: {entangled}")
    
    return entangled

# Test with current observation
print("\nTesting quantum heuristic...")
entangled = quantum_filter_actions(obs, [], corr_matrix)

# ── Step 6: Save everything ──
torch.save({
    'gnn_state': gnn.state_dict(),
    'edge_index': edge_index,
    'corr_matrix': corr_matrix,
    'node_feature_dim': 3,
    'embedding_dim': 64
}, 'gnn_encoder.pt')

print("\n✅ GNN encoder saved to gnn_encoder.pt")
print("✅ Quantum heuristic ready")
print("\nYour AI stack is complete:")
print("  PPO Agent    — routing decisions")
print("  LSTM         — demand forecasting")  
print("  GNN          — topology understanding")
print("  Quantum filter — smart action selection")