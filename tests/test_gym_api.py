# test_gym_api.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json
import gymnasium as gym

init_meta = {"exp_code": "gym_test", "exp_id": 9, "exp_name": "gym", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
env = ENV_BESS(**config)

print("Checking Gym API compliance...\n")

# Check inheritance
assert isinstance(env, gym.Env), "Must inherit from gym.Env"
print("[PASS] Inherits from gym.Env")

# Check spaces
assert hasattr(env, 'action_space'), "Missing action_space"
assert hasattr(env, 'observation_space'), "Missing observation_space"
print("[PASS] Has action_space and observation_space")

# Check reset signature
obs, info = env.reset()
assert isinstance(obs, dict), "Observation must be dict"
assert isinstance(info, dict), "Info must be dict"
print("[PASS] reset() returns (observation, info)")

# Check step signature
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
assert isinstance(reward, (int, float)), "Reward must be number"
assert isinstance(terminated, bool), "terminated must be bool"
assert isinstance(truncated, bool), "truncated must be bool"
print("[PASS] step() returns (obs, reward, terminated, truncated, info)")

print("\n[PASS] Gym API compliance verified")
