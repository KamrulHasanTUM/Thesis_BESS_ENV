from stable_baselines3 import PPO
from config import load_config, create_bess_env_config
from ENV_BESS_main import ENV_BESS

cfg = create_bess_env_config(load_config())
env = ENV_BESS(**cfg)
model = PPO.load("final_model", env=env)

obs, info = env.reset()
for _ in range(10):
    action, _ = model.predict(obs, deterministic=True)
    print("PPO action (MW):", action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()