"""
improved_env_helpers.py

Contains improved versions of key functions:
1. SoC-aware action clipping
2. Simplified reward function
3. Fixed action scaling

These improvements prevent physically impossible actions and simplify learning.
"""

import numpy as np


def apply_bess_action_with_soc_clipping(env, action_normalized):
    """
    Apply BESS action with SoC-aware clipping to prevent impossible operations.

    This function ensures agent cannot:
    - Discharge when battery is empty
    - Charge when battery is full
    - Violate SoC constraints

    Args:
        env: Environment instance with BESS state
        action_normalized: Normalized action [-1, +1] from agent

    Returns:
        tuple: (clipped_action_mw, clip_info)
            - clipped_action_mw: Action in MW after SoC-aware clipping
            - clip_info: Dictionary with clipping statistics
    """
    # Scale normalized action to MW
    action_mw = action_normalized * env.bess_power_mw

    # Storage for clipped actions and info
    clipped_action = np.zeros_like(action_mw)
    clip_info = {
        'units_clipped': [],
        'original_actions': [],
        'clipped_actions': [],
        'clip_reasons': []
    }

    for i in range(env.num_bess):
        current_soc = env.bess_soc[i]
        requested_action = action_mw[i]

        # Calculate energy constraints
        # Maximum energy available for discharge (MWh)
        available_energy = (current_soc - env.soc_min) * env.bess_capacity_mwh

        # Maximum power for discharge given available energy (MW)
        max_discharge_power = available_energy / env.time_step_hours

        # Maximum energy space for charging (MWh)
        available_space = (env.soc_max - current_soc) * env.bess_capacity_mwh

        # Maximum power for charging given available space (MW)
        max_charge_power = available_space / env.time_step_hours

        # Apply constraints
        if requested_action > 0:  # Discharging (positive power)
            # Limit by: available energy, power rating
            feasible_discharge = min(max_discharge_power, env.bess_power_mw)
            clipped_action[i] = min(requested_action, feasible_discharge)

            if clipped_action[i] < requested_action:
                clip_info['units_clipped'].append(i)
                clip_info['original_actions'].append(float(requested_action))
                clip_info['clipped_actions'].append(float(clipped_action[i]))
                clip_info['clip_reasons'].append(f"BESS {i}: Insufficient energy (SoC={current_soc:.1%})")

        elif requested_action < 0:  # Charging (negative power)
            # Limit by: available space, power rating
            feasible_charge = min(max_charge_power, env.bess_power_mw)
            clipped_action[i] = max(requested_action, -feasible_charge)

            if clipped_action[i] > requested_action:  # Less negative = clipped
                clip_info['units_clipped'].append(i)
                clip_info['original_actions'].append(float(requested_action))
                clip_info['clipped_actions'].append(float(clipped_action[i]))
                clip_info['clip_reasons'].append(f"BESS {i}: Insufficient space (SoC={current_soc:.1%})")

        else:  # Idle (zero action)
            clipped_action[i] = 0.0

    return clipped_action, clip_info


def calculate_simplified_reward(env, max_loading_before, max_loading_after):
    """
    Simplified reward function - ONLY congestion reduction.

    Removes all other components (action magnitude bonus, efficiency penalties)
    to isolate whether agent can learn the core task.

    Args:
        env: Environment instance
        max_loading_before: Maximum line loading before action (%)
        max_loading_after: Maximum line loading after action (%)

    Returns:
        tuple: (total_reward, reward_breakdown)
    """
    # Pure congestion reduction
    delta_loading = max_loading_before - max_loading_after

    # Scale to reasonable range (typical delta is 0-5%, scale to ±50)
    congestion_reward = delta_loading * 10.0

    # Clip to prevent extreme values
    congestion_reward = np.clip(congestion_reward, -500, 500)

    reward_breakdown = {
        'congestion': float(congestion_reward),
        'delta_loading': float(delta_loading)
    }

    return congestion_reward, reward_breakdown


def calculate_improved_reward(env, max_loading_before, max_loading_after):
    """
    Improved reward function with balanced components.

    Components:
    1. Congestion reduction (primary)
    2. SoC boundary penalties (secondary, but stronger)
    3. NO action magnitude bonus (removed)

    Args:
        env: Environment instance
        max_loading_before: Maximum line loading before action (%)
        max_loading_after: Maximum line loading after action (%)

    Returns:
        tuple: (total_reward, reward_breakdown)
    """
    # ===== 1. Congestion Reduction (Primary Objective) =====
    bonus_constant = getattr(env, 'bonus_constant', 100.0)
    delta_loading = max_loading_before - max_loading_after
    congestion_reward = bonus_constant * delta_loading

    # ===== 2. SoC Boundary Penalties (Prevent Constraint Violations) =====
    soc_penalty_weight = getattr(env, 'soc_penalty_weight', -5.0)  # Increased from -0.2
    boundary_margin = getattr(env, 'soc_boundary_margin', 0.05)

    # Count units near SoC bounds
    num_near_lower = np.sum(env.bess_soc < (env.soc_min + boundary_margin))
    num_near_upper = np.sum(env.bess_soc > (env.soc_max - boundary_margin))
    num_near_bounds = num_near_lower + num_near_upper

    soc_penalty = soc_penalty_weight * num_near_bounds

    # ===== Total Reward =====
    total_reward = congestion_reward + soc_penalty

    reward_breakdown = {
        'congestion': float(congestion_reward),
        'soc_penalty': float(soc_penalty),
        'delta_loading': float(delta_loading),
        'num_near_bounds': int(num_near_bounds)
    }

    return total_reward, reward_breakdown


def validate_action_bounds(action_normalized):
    """
    Validate that actions are within expected normalized bounds [-1, +1].

    Args:
        action_normalized: Action array from agent

    Returns:
        tuple: (is_valid, issues)
            - is_valid: Boolean, True if all actions in bounds
            - issues: List of issues found
    """
    issues = []
    is_valid = True

    for i, action in enumerate(action_normalized):
        if action < -1.0:
            issues.append(f"BESS {i}: Action {action:.3f} below -1.0")
            is_valid = False
        elif action > 1.0:
            issues.append(f"BESS {i}: Action {action:.3f} above +1.0")
            is_valid = False

    return is_valid, issues


# Diagnostic utilities

def analyze_action_magnitude_vs_loading(loading_before, action_magnitude):
    """
    Analyze relationship between loading level and action magnitude.

    Helps identify if agent is scaling actions appropriately to congestion severity.

    Args:
        loading_before: Loading percentage before action
        action_magnitude: Magnitude of action taken (MW)

    Returns:
        dict: Analysis results
    """
    # Expected action magnitude based on loading
    if loading_before > 80:
        expected_range = (30, 50)
        category = "high"
    elif loading_before > 50:
        expected_range = (20, 40)
        category = "medium"
    elif loading_before > 30:
        expected_range = (10, 25)
        category = "low-medium"
    else:
        expected_range = (0, 15)
        category = "low"

    is_appropriate = expected_range[0] <= action_magnitude <= expected_range[1]

    return {
        'loading_before': loading_before,
        'action_magnitude': action_magnitude,
        'expected_range': expected_range,
        'category': category,
        'is_appropriate': is_appropriate
    }
