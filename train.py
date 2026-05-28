import grid2op
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from grid2op.Reward import L2RPNReward
import gymnasium as gym

# Load environment
env_g2op = grid2op.make("l2rpn_case14_sandbox", reward_class=L2RPNReward)

# Custom wrapper that makes do-nothing the default
class SafeGridEnv(gym.Env):
    def __init__(self, g2op_env):
        self.g2op_env = g2op_env
        self.obs_size = 6 * g2op_env.n_line  # rho + 5 other features per line
        
        self.observation_space = gym.spaces.Box(
            low=-1, high=2, 
            shape=(self.obs_size,), 
            dtype=np.float32
        )
        # Only 3 actions: do nothing, or two simple redispatch actions
        self.action_space = gym.spaces.Discrete(3)
        self.current_obs = None
        
    def _get_obs(self, obs):
        return np.concatenate([
            obs.rho,           # line load ratios
            obs.p_or,          # power flow origin
            obs.p_ex,          # power flow extremity  
            obs.v_or,          # voltage origin
            obs.a_or,          # current origin
            obs.line_status.astype(float)  # line on/off
        ]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        obs = self.g2op_env.reset()
        self.current_obs = obs
        return self._get_obs(obs), {}
    
    def step(self, action):
        # Action 0 = do nothing (safest)
        # Action 1 = reconnect any disconnected line
        # Action 2 = do nothing (double weight on safe action)
        if action == 1:
            g2op_action = self.g2op_env.action_space(
                {"set_line_status": [(0, 1)]}  # reconnect line 0
            )
        else:
            g2op_action = self.g2op_env.action_space({})  # do nothing
            
        obs, reward, done, info = self.g2op_env.step(g2op_action)
        self.current_obs = obs
        
        # Custom reward: reward survival, penalize high line loads
        if done:
            custom_reward = -50.0
        else:
            survival = 1.0
            overload_penalty = -5.0 * np.sum(obs.rho > 0.9)
            loss_penalty = -0.01 * np.sum(obs.p_or - obs.p_ex)
            custom_reward = survival + overload_penalty + loss_penalty
            
        return self._get_obs(obs), custom_reward, done, False, info

# Create safe environment
safe_env = SafeGridEnv(env_g2op)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="gridmind_ppo"
)

# Train PPO
model = PPO(
    "MlpPolicy",
    safe_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01
)

print("Starting training with safe action space...")
print("ep_len_mean should climb well above 8 now")

model.learn(total_timesteps=50_000, callback=checkpoint_callback)
model.save("gridmind_ppo_final")
print("Done! Model saved.")