# test_reset.py
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config
import json
import numpy as np

init_meta = {"exp_code": "reset_test", "exp_id": 3, "exp_name": "reset", "grid_env": "bess"}
config = create_bess_env_config(init_meta)
env = ENV_BESS(**config)

print("Testing reset()...")
obs, info = env.reset()

# Verify observation structure
assert isinstance(obs, dict), "Observation should be dict"
assert 'bess_soc' in obs, "Missing bess_soc"
assert 'bess_power' in obs, "Missing bess_power"
assert obs['bess_soc'].shape == (5,), f"Wrong SoC shape: {obs['bess_soc'].shape}"
assert obs['bess_power'].shape == (5,), f"Wrong power shape: {obs['bess_power'].shape}"

# Verify initial values
assert np.allclose(obs['bess_soc'], 0.5), "Initial SoC should be 0.5"
assert np.allclose(obs['bess_power'], 0.0), "Initial power should be 0"

print("[PASS] Reset test passed")
print(f"  Observation keys: {list(obs.keys())}")
print(f"  Initial SoC: {obs['bess_soc']}")
print(f"  Initial power: {obs['bess_power']}")
