# test_full_episode.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json
import numpy as np

init_meta = {"exp_code": "full_ep", "exp_id": 7, "exp_name": "full", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
config['max_step'] = 20  # Shorter for testing

env = ENV_BESS(**config)

print(f"Running full episode (max {config['max_step']} steps)...\n")
obs, _ = env.reset()
total_reward = 0
step_count = 0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    step_count += 1
    
    if terminated or truncated:
        print(f"Episode finished after {step_count} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        break

assert step_count <= config['max_step'], "Episode exceeded max steps"
print("\n[PASS] Full episode test passed")
