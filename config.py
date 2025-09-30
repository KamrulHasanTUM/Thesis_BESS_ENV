"""
config.py

Configuration management functions for loading and creating experiment configurations.
Handles loading from JSON files and creating environment and training configurations.
"""

import json


def load_config():
    """Load configuration from init_meta.json file."""
    with open("init_meta.json", "r") as file:
        init_meta = json.load(file)
    import pdb; pdb.set_trace()
    print(init_meta)
    return init_meta


def create_env_config(init_meta):
    """Create environment configuration parameters."""
    return {
        'simbench_code': "1-HV-mixed--0-sw",
        'case_study': 'bc',
        'is_train': True,
        'is_normalize': False,
        'max_step': 50,
        'allowed_lines': 100,
        'convergence_penalty': -200,
        'line_disconnect_penalty': -200,
        'nan_vm_pu_penalty': "dynamic",
        'rho_min': 0.45,
        'penalty_scalar': -10,
        'bonus_constant': 10,
        'action_type': init_meta["action_type"],
        'exp_code': init_meta["exp_code"]
    }


def create_training_config(init_meta):
    """Create training configuration parameters."""
    return {
        'n_epochs': 10,
        'n_steps': 2048,
        'batch_size': 256,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'total_timesteps': 1_000_000,
        'initial_learning_rate': 0.0003,
        'exp_id': init_meta["exp_id"],
        'exp_code': init_meta["exp_code"],
        'exp_name': init_meta["exp_name"],
        'grid_env': init_meta["grid_env"]
    }


def save_training_metadata(training_config, logdir):
    """Save training configuration metadata to JSON file."""
    training_config_meta = {
        "exp_id": training_config['exp_id'],
        "exp_code": training_config['exp_code'],
        "exp_name": training_config['exp_name'],
        "logdir": logdir,
        "env_name": f"ENV_{training_config['grid_env']}",
        "policy": "MultiInputPolicy",
        "n_epochs": training_config['n_epochs'],
        "n_steps": training_config['n_steps'],
        "batch_size": training_config['batch_size'],
        "gamma": training_config['gamma'],
        "gae_lambda": training_config['gae_lambda'],
        "clip_range": training_config['clip_range'],
        "ent_coef": training_config['ent_coef'],
        "max_grad_norm": training_config['max_grad_norm'],
        "total_timesteps": training_config['total_timesteps'],
        "initial_learning_rate": training_config['initial_learning_rate']
    }

    with open("training_config_meta.json", "w") as file:
        json.dump(training_config_meta, file, indent=4)
    print("Data saved to training_config_meta.json")
