# test_multiple_episodes.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json

init_meta = {"exp_code": "multi_ep", "exp_id": 8, "exp_name": "multi", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
config['max_step'] = 10

env = ENV_BESS(**config)

print("Running 5 consecutive episodes...\n")

for ep in range(5):
    obs, _ = env.reset()
    steps = 0
    
    while steps < config['max_step']:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"Episode {ep+1}: {steps} steps completed")

print("\n[PASS] Multiple episodes test passed - no crashes")
