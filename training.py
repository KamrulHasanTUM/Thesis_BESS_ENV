"""
training.py

Training pipeline functions for setting up environment, creating models, and training.
Handles environment setup, PPO model creation, and training execution with callbacks.
"""

import os
import datetime
from wandb_integration import WandbCallback, init_wandb_run
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

from utils import TimeStepLoggingCallback, TQDMProgressCallback


def setup_environment(env_class, env_config):
    """
    Setup and wrap the environment.

    Args:
        env_class: The environment class to instantiate
        env_config: Dictionary of environment configuration parameters

    Returns:
        DummyVecEnv: Wrapped environment ready for training
    """
    monitored_env = Monitor(env_class(**env_config))
    return DummyVecEnv([lambda: monitored_env])


def create_model(env, training_config, logdir):
    """
    Create and configure PPO model.

    Args:
        env: The vectorized environment
        training_config: Dictionary of training configuration parameters
        logdir: Directory for tensorboard logs

    Returns:
        PPO: Configured PPO model
    """
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        tensorboard_log=logdir,
        n_epochs=training_config['n_epochs'],
        n_steps=training_config['n_steps'],
        batch_size=training_config['batch_size'],
        gamma=training_config['gamma'],
        gae_lambda=training_config['gae_lambda'],
        clip_range=training_config['clip_range'],
        ent_coef=training_config['ent_coef'],
        max_grad_norm=training_config['max_grad_norm']
    )


def train_model(model, training_config, tqdm_callback):
    """
    Train the model with callbacks and handle interruptions.

    Args:
        model: The PPO model to train
        training_config: Dictionary of training configuration parameters
        tqdm_callback: TQDM progress bar callback
    """
    callback_list = CallbackList([TimeStepLoggingCallback(), tqdm_callback])

    try:
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            tb_log_name=f"{training_config['exp_code']}",
            reset_num_timesteps=False,
            callback=callback_list
        )
        print("Training Ends")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Closing progress bar and saving model...")
    finally:
        if hasattr(tqdm_callback, 'pbar') and tqdm_callback.pbar is not None:
            tqdm_callback.pbar.close()

        print(f"Saving model to {training_config['exp_code']}")
        model.save(f"{training_config['exp_code']}")
        print("Model saved successfully!")

def train_model_with_wandb(model, training_config, env_config):
    """Train model with W&B tracking."""
    full_config = {**env_config, **training_config}
    run = init_wandb_run(full_config, project_name="thesis-bess-env")
    
    wandb_callback = WandbCallback(log_freq=10)
    
    try:
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=wandb_callback,
            progress_bar=True
        )
        
        model.save("final_model")
        artifact = wandb.Artifact('bess-ppo-model', type='model')
        artifact.add_file('final_model.zip')
        run.log_artifact(artifact)
    finally:
        run.finish()


def get_logdir():
    """
    Generate a timestamped log directory path.

    Returns:
        str: Path to log directory
    """
    return os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
