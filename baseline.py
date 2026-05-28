import grid2op
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # silence the pandas warnings

env = grid2op.make("l2rpn_case14_sandbox")
obs = env.reset()

total_loss = []
done = False
step = 0

while not done and step < 200:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    loss = np.sum(obs.p_or - obs.p_ex)
    total_loss.append(loss)
    step += 1

baseline_loss = np.mean(total_loss)
print(f"Steps survived: {step}")
print(f"Baseline average transmission loss: {baseline_loss:.2f} MW")
print(f">>> WRITE THIS NUMBER DOWN: {baseline_loss:.2f} MW <<<")