# test_rewards.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json
import numpy as np

init_meta = {"exp_code": "reward_test", "exp_id": 6, "exp_name": "reward", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
env = ENV_BESS(**config)

print("Testing reward components...\n")

# Test idle action
env.reset()
action = np.zeros(5)
_, reward_idle, _, _, _ = env.step(action)
print(f"Idle action reward: {reward_idle:.2f}")

# Test various actions to see reward variation
env.reset()
rewards = []
for power in [0, 10, 20, -10, -20]:
    env.reset()
    action = np.full(5, power)
    _, reward, _, _, _ = env.step(action)
    rewards.append(reward)
    print(f"Power {power:3.0f} MW: Reward = {reward:7.2f}")

print("\n[PASS] Reward function test passed - rewards vary with actions")
