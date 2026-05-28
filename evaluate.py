import grid2op
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import gymnasium as gym
from stable_baselines3 import PPO

# Same environment setup as training
env_g2op = grid2op.make("l2rpn_case14_sandbox")

class SafeGridEnv(gym.Env):
    def __init__(self, g2op_env):
        self.g2op_env = g2op_env
        self.obs_size = 6 * g2op_env.n_line
        self.observation_space = gym.spaces.Box(
            low=-1, high=2,
            shape=(self.obs_size,),
            dtype=np.float32
        )
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
                {"set_line_status": [(0, 1)]}
            )
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

# Load trained model
safe_env = SafeGridEnv(env_g2op)
model = PPO.load("gridmind_ppo_final", env=safe_env)

print("Evaluating GridMind vs Baseline...")
print("=" * 40)

# Run GridMind for 200 steps
total_loss = []
obs, _ = safe_env.reset()
for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = safe_env.step(action)
    g2op_obs = env_g2op.current_obs
    loss = np.sum(g2op_obs.p_or - g2op_obs.p_ex)
    total_loss.append(loss)
    if done:
        obs, _ = safe_env.reset()

gridmind_loss = np.mean(total_loss)
baseline_loss = 562.14  # your recorded number

improvement = ((baseline_loss - gridmind_loss) / baseline_loss) * 100

print(f"Baseline loss:  {baseline_loss:.2f} MW")
print(f"GridMind loss:  {gridmind_loss:.2f} MW")
print(f"Improvement:    {improvement:.1f}%")
print("=" * 40)
print(f">>> YOUR HEADLINE NUMBER: {improvement:.1f}% reduction in transmission loss <<<")