import grid2op
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import gymnasium as gym
from stable_baselines3 import PPO

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

safe_env = SafeGridEnv(env_g2op)
model = PPO.load("gridmind_ppo_final", env=safe_env)

print("Evaluating GridMind vs Baseline...")
print("=" * 45)

# ── GridMind: 5 episodes ──────────────────────
num_episodes = 5
all_episode_losses = []
all_episode_steps = []

for ep in range(num_episodes):
    obs, _ = safe_env.reset()
    ep_loss = 0
    steps = 0
    done = False

    while not done and steps < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = safe_env.step(action)
        g2op_obs = env_g2op.current_obs
        step_loss = np.sum(g2op_obs.p_or - g2op_obs.p_ex)
        ep_loss += step_loss
        steps += 1

    all_episode_losses.append(ep_loss / steps)
    all_episode_steps.append(steps)
    print(f"Episode {ep+1}: {steps} steps survived, avg loss = {ep_loss/steps:.2f} MW")

gridmind_loss = np.mean(all_episode_losses)
gridmind_steps = np.mean(all_episode_steps)

# ── Baseline: same 5 episodes with do-nothing ─
print("\nRunning baseline...")
baseline_losses = []
baseline_steps_list = []

for ep in range(num_episodes):
    obs = env_g2op.reset()
    ep_loss = 0
    steps = 0
    done = False

    while not done and steps < 200:
        action = env_g2op.action_space({})  # do nothing
        obs, reward, done, info = env_g2op.step(action)
        step_loss = np.sum(obs.p_or - obs.p_ex)
        ep_loss += step_loss
        steps += 1

    baseline_losses.append(ep_loss / steps)
    baseline_steps_list.append(steps)
    print(f"Episode {ep+1}: {steps} steps survived, avg loss = {ep_loss/steps:.2f} MW")

baseline_loss = np.mean(baseline_losses)
baseline_steps = np.mean(baseline_steps_list)

# ── Final comparison ──────────────────────────
improvement = ((baseline_loss - gridmind_loss) / baseline_loss) * 100
step_improvement = ((gridmind_steps - baseline_steps) / baseline_steps) * 100

print("\n" + "=" * 45)
print(f"Baseline  — Avg loss: {baseline_loss:.2f} MW | Avg survival: {baseline_steps:.0f} steps")
print(f"GridMind  — Avg loss: {gridmind_loss:.2f} MW | Avg survival: {gridmind_steps:.0f} steps")
print(f"\n>>> LOSS IMPROVEMENT:     {improvement:.1f}% <<<")
print(f">>> SURVIVAL IMPROVEMENT: {step_improvement:.1f}% <<<")
print("=" * 45)
print("\nThese are your headline numbers for the demo!")