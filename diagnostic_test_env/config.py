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
    print(init_meta)
    return init_meta

def create_bess_env_config(init_meta):
    """
    Create BESS environment configuration parameters.

    This function extends create_env_config() by adding BESS-specific parameters
    for battery energy storage system modeling while preserving all grid environment
    parameters (penalties, network settings, episode configuration).

    Differences from create_env_config():
    - Adds BESS unit specifications (number, capacity, power rating)
    - Adds BESS operational constraints (SoC limits, efficiency)
    - Adds time parameters for energy accounting (time_step_hours)

    BESS Parameters Explanation:
    - num_bess: Number of BESS units deployed in the grid
    - bess_capacity_mwh: Energy storage capacity per unit (MWh)
    - bess_power_mw: Maximum charge/discharge power per unit (MW)
    - soc_min/max: State of charge operating window (10-90% prevents degradation)
    - initial_soc: Starting energy level (50% allows bidirectional flexibility)
    - efficiency: Round-trip energy efficiency (90% is typical for Li-ion)
    - time_step_hours: Timestep duration for SoC calculations (Energy = Power × Time)

    Example Usage:
        >>> init_meta = load_config()
        >>> bess_config = create_bess_env_config(init_meta)
        >>> env = ENV_BESS(**bess_config)

    Returns:
        dict: Complete configuration dictionary with both grid and BESS parameters
    """
    return {
        # ========== Grid Environment Parameters (from ENV_RHV) ==========
        'simbench_code': "1-HV-mixed--0-sw",
        'case_study': 'hL',
        'is_train': True,
        'is_normalize': False,
        'max_step': 50,
        'allowed_lines': 100,
        'convergence_penalty': -50,
        'line_disconnect_penalty': -50,
        'nan_vm_pu_penalty': -20,
        'penalty_scalar': -10,
        'bonus_constant': 100,  # Reduced from 50000 to make rewards learnable
        'exp_code': init_meta["exp_code"],

        # ========== BESS Unit Configuration ==========
        # Number of BESS units to deploy in the grid
        'num_bess': 5,

        # GA-OPTIMIZED BESS LOCATIONS (from genetic algorithm)
        # Fitness Score: 62.92 (optimal congestion reduction)
        'bess_locations': [39, 189, 230, 281, 282],

        # Energy capacity per BESS unit (MWh)
        # Determines how much energy each battery can store
        'bess_capacity_mwh': 50.0,

        # Power rating per BESS unit (MW)
        # Maximum rate at which the battery can charge or discharge
        # Action space will be continuous: [-bess_power_mw, +bess_power_mw]
        'bess_power_mw': 50.0,

        # ========== BESS Operating Constraints ==========
        # Minimum state of charge (10%)
        # Prevents deep discharge which degrades battery lifespan
        # Also maintains reserve capacity for emergency response
        'soc_min': 0.1,

        # Maximum state of charge (90%)
        # Prevents overcharging which can damage cells
        # Also maintains headroom for absorbing excess renewable generation
        'soc_max': 0.9,

        # Initial state of charge (50%)
        # Starting at mid-range allows flexibility to charge or discharge
        # Represents a neutral, ready-to-respond state
        'initial_soc': 0.5,

        # Round-trip efficiency (90%)
        # Energy loss during charge/discharge cycle
        # Typical for modern Li-ion batteries (85-95%)
        # Applied as: energy_stored = power_in * efficiency * time
        'efficiency': 0.9,

        # ========== Time Parameters ==========
        # Duration of each simulation timestep (hours)
        # Used for SoC calculations: delta_SoC = (P_mw * time_step_hours) / capacity_mwh
        # 1.0 hour is common for grid-scale BESS planning
        'time_step_hours': 1.0,
        # Engineering constants
        'voltage_min_pu': 0.5,
        'voltage_max_pu': 1.5,
        'soc_boundary_margin': 0.05,
        # After line 105 ('soc_boundary_margin': 0.05), ADD:
        'soc_penalty_weight': -0.2  # Reduced from -1.0 to allow more freedom
    
    }


def create_training_config(init_meta):
    """Create training configuration parameters."""
    return {
        'n_epochs': 10,
        'n_steps': 2048,
        'batch_size': 256,
        'gamma': 0.99,
        'gae_lambda': 0.95,

        # ========== PPO HYPERPARAMETER TUNING (Section 12.6) ==========
        # BEFORE: clip_range = 0.2
        # AFTER: clip_range = 0.3
        # Why: Larger clip range can handle reward variance better
        #      With reward range of -532 to +435, clip_range=0.2 may be too conservative
        #      Allows larger policy updates while still preventing catastrophic changes
        'clip_range': 0.3,

        'ent_coef': 0.10,
        'max_grad_norm': 0.5,

        # ========== DIAGNOSTIC TEST CONFIGURATION ==========
        # DIAGNOSTIC: total_timesteps = 5,000
        # Why: Quick diagnostic test to verify improvements work
        #      5K timesteps = ~10-15 minutes of training
        #      Target: <20% harmful actions, >80% success rate
        #      If successful, scale up to 50K for full training
        'total_timesteps': 5_000,

        # ========== LEARNING RATE TUNING (Section 12.6) ==========
        # CURRENT: initial_learning_rate = 0.0003
        # This is already the recommended value from Section 12.6
        # Why: Slightly faster learning than 0.0002
        #      Faster convergence when combined with reward normalization
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
