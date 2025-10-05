# test_env_creation.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json

init_meta = {
    "exp_code": "env_test",
    "exp_id": 2,
    "exp_name": "env_creation",
    "grid_env": "bess"
}

config = create_bess_env_config(init_meta)

print("Creating environment...")
env = ENV_BESS(**config)

print("[PASS] Environment created successfully")
print(f"  Action space: {env.action_space}")
print(f"  Observation space keys: {list(env.observation_space.spaces.keys())}")
print(f"  BESS units: {env.num_bess}")
print(f"  BESS capacity: {env.bess_capacity_mwh} MWh")
print(f"  BESS power: {env.bess_power_mw} MW")
