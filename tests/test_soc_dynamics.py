# test_soc_dynamics.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json
import numpy as np

init_meta = {"exp_code": "soc_test", "exp_id": 5, "exp_name": "soc", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
env = ENV_BESS(**config)

obs, _ = env.reset()
initial_soc = obs['bess_soc'].copy()

print("Test A: Discharge should decrease SoC")
action = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
obs, reward, _, _, _ = env.step(action)
print(f"  Initial SoC: {initial_soc[0]:.3f}")
print(f"  After discharge: {obs['bess_soc'][0]:.3f}")
assert obs['bess_soc'][0] < initial_soc[0], "SoC should decrease after discharge"
print("  [PASS] Discharge decreases SoC\n")

env.reset()
obs, _ = env.reset()
initial_soc = obs['bess_soc'].copy()

print("Test B: Charge should increase SoC")
action = np.array([-20.0, -20.0, -20.0, -20.0, -20.0])
obs, reward, _, _, _ = env.step(action)
print(f"  Initial SoC: {initial_soc[0]:.3f}")
print(f"  After charge: {obs['bess_soc'][0]:.3f}")
assert obs['bess_soc'][0] > initial_soc[0], "SoC should increase after charge"
print("  [PASS] Charge increases SoC\n")

print("[PASS] SoC dynamics test passed")
