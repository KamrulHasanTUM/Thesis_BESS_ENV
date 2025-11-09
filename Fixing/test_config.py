"""
test_config.py

Configuration for diagnostic tests - uses parent directory modules
"""

import sys
import os

# Add parent directory to path to import original modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import load_config, create_bess_env_config, create_training_config

def get_test_env_config():
    """Get environment configuration for testing."""
    init_meta = {
        "exp_code": "diagnostic_test",
        "exp_id": 999,
        "exp_name": "baseline_diagnostic",
        "grid_env": "bess"
    }

    return create_bess_env_config(init_meta)


def get_fast_test_env_config():
    """Get faster environment config for quick testing (reduced episode length)."""
    config = get_test_env_config()
    config['max_step'] = 10  # Reduced from 50 for faster testing
    return config


def get_improved_env_config():
    """Get improved environment config with fixes applied."""
    config = get_test_env_config()

    # Apply improvements
    config['max_step'] = 10  # Reduced episode length for faster learning
    config['soc_penalty_weight'] = -5.0  # Increased from -0.2 for stronger constraint

    return config


# Test parameters
TEST_PARAMS = {
    'num_episodes': 20,  # Number of episodes for baseline tests (reduced for faster testing)
    'max_steps_per_episode': 50,  # Steps per episode
    'random_seed': 42,  # For reproducibility
    'results_dir': os.path.join(os.path.dirname(__file__), 'results')
}
