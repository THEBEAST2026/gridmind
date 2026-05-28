import grid2op
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env = grid2op.make("l2rpn_case14_sandbox")
obs = env.reset()

print("Testing do-nothing action for 20 steps...")
for i in range(20):
    # Do nothing action - safest possible action
    action = env.action_space({})
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.2f}, done={done}, reason={info.get('exception', 'none')}")
    if done:
        print("Environment ended early!")
        print("Reason:", info)
        break