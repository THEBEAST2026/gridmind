import grid2op

env = grid2op.make("l2rpn_case14_sandbox")
obs = env.reset()

print("Environment loaded!")
print("Number of lines:", env.n_line)
print("Number of substations:", env.n_sub)
print("Observation shape:", obs.to_vect().shape)