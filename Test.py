from config import load_config, create_bess_env_config
from ENV_BESS_main import ENV_BESS
cfg = create_bess_env_config(load_config())
env = ENV_BESS(**cfg)
obs, info = env.reset()
for step in range(10):
    obs, reward, terminated, truncated, info = env.step([0.0]*env.num_bess)
    if terminated or truncated:
        break
