# test_config.py
from config import load_config, create_bess_env_config, create_training_config
import json

# Create minimal init_meta.json first
init_meta = {
    "exp_code": "verification_test",
    "exp_id": 1,
    "exp_name": "full_verification",
    "grid_env": "bess"
}

with open("init_meta.json", "w") as f:
    json.dump(init_meta, f)

# Test loading
loaded = load_config()
print("Config loaded:", loaded)

# Test BESS config
bess_config = create_bess_env_config(loaded)
print("BESS config keys:", bess_config.keys())
assert bess_config['num_bess'] == 5
assert bess_config['bess_power_mw'] == 50.0
print("[PASS] Configuration test passed")
