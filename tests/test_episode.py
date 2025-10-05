# test_episode.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json
import numpy as np

init_meta = {"exp_code": "episode_test", "exp_id": 4, "exp_name": "episode", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
env = ENV_BESS(**config)

print("Running 10-step episode with random actions...\n")
obs, info = env.reset()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step+1}:")
    print(f"  Action: {action[:3]}... (showing first 3)")
    print(f"  Reward: {reward:.2f}")
    print(f"  SoC: {obs['bess_soc']}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}\n")
    
    if terminated or truncated:
        print("Episode ended early")
        break

print("[PASS] Episode test passed - no crashes")
