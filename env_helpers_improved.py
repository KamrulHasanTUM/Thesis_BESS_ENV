"""
IMPROVED VERSION - env_helpers with fixes for policy collapse

This is a TESTING version that includes fixes for the training collapse issue.
It will be used ONLY in diagnostic tests to verify improvements work.

CHANGES FROM ORIGINAL:
1. Added baseline_penalty to penalize inaction
2. Added action_utilization_bonus to reward taking action
3. Added diversity_bonus to encourage using multiple units
4. Modified flexibility_bonus to penalize stuck state
5. All other functions remain IDENTICAL to original

DO NOT USE THIS IN PRODUCTION - FOR TESTING ONLY
"""

import sys
import os

# Import all functions from original env_helpers
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import everything from original env_helpers
from env_helpers import *

# Override ONLY the reward calculation function
def calculate_bess_reward(env, max_loading_before, max_loading_after):
    """
    IMPROVED reward calculation that prevents 'do nothing' policy collapse.

    Adds three new components:
    1. baseline_penalty: Strong penalty for inaction
    2. action_utilization_bonus: Reward for using BESS capacity
    3. diversity_bonus: Reward for using multiple units

    Also modifies flexibility_bonus to penalize stuck state.
    """

    # Get the last action from environment (if available)
    # This is set by the step function before calling reward calculation
    action = getattr(env, '_last_action', np.zeros(env.num_bess))

    # ===== ORIGINAL COMPONENTS (UNCHANGED) =====

    # 1. Congestion reward
    bonus_constant = getattr(env, 'bonus_constant', 10.0)
    congestion_reward = bonus_constant * (max_loading_before - max_loading_after)

    # 2. SoC penalty
    soc_penalty_weight = getattr(env, 'soc_penalty_weight', -1.0)
    boundary_margin = getattr(env, 'soc_boundary_margin', 0.05)

    num_near_lower = np.sum(env.bess_soc < (env.soc_min + boundary_margin))
    num_near_upper = np.sum(env.bess_soc > (env.soc_max - boundary_margin))
    num_near_bounds = num_near_lower + num_near_upper
    soc_penalty = soc_penalty_weight * num_near_bounds

    # 3. Flexibility bonus (MODIFIED to penalize stuck state)
    flex_weight = getattr(env, 'flexibility_bonus_weight', 0.0)
    flex_center = getattr(env, 'soc_flex_center', 0.5)
    flex_width = getattr(env, 'soc_flex_width', 0.2)

    flexibility_bonus = 0.0
    if flex_weight > 0 and flex_width > 0:
        distance = np.abs(env.bess_soc - flex_center)
        in_band = distance <= flex_width
        if np.any(in_band):
            normalized_distance = distance[in_band] / flex_width
            flexibility_bonus = flex_weight * np.sum(1.0 - normalized_distance)

        # NEW: Penalize stuck at center with no action
        if np.allclose(env.bess_soc, 0.5, atol=0.02) and np.all(np.abs(action) < 0.05):
            flexibility_bonus -= 20.0  # Strong penalty for being stuck

    # 4. Clipping penalty
    clipping_penalty_weight = getattr(env, 'soc_clipping_penalty', 0.0)
    clip_info = env.info.get('clip_info', {'units_clipped': []})
    num_clipped = len(clip_info['units_clipped'])
    clipping_penalty = clipping_penalty_weight * num_clipped

    # ===== NEW COMPONENTS (PREVENT POLICY COLLAPSE) =====

    # 5. NEW: Baseline penalty (penalize inaction)
    baseline_penalty = 0.0
    if np.all(np.abs(action) < 0.01):  # Essentially doing nothing
        baseline_penalty = -50.0  # Strong penalty for complete inaction

    # 6. NEW: Action utilization bonus (reward taking action)
    action_magnitude = np.mean(np.abs(action))
    action_utilization_bonus = 5.0 * action_magnitude  # 0 to +5

    # 7. NEW: Diversity bonus (reward using multiple units)
    num_active = np.sum(np.abs(action) > 0.1)  # Count units with significant action
    diversity_bonus = 2.0 * num_active  # 0 to +10 (2 per active unit)

    # ===== TOTAL REWARD =====

    total_reward = (congestion_reward +
                   soc_penalty +
                   flexibility_bonus +
                   clipping_penalty +
                   baseline_penalty +
                   action_utilization_bonus +
                   diversity_bonus)

    # ===== REWARD BREAKDOWN (for logging) =====

    reward_breakdown = {
        'congestion': float(congestion_reward),
        'soc_penalty': float(soc_penalty),
        'num_near_bounds': int(num_near_bounds),
        'flexibility_bonus': float(flexibility_bonus),
        'clipping_penalty': float(clipping_penalty),
        'num_clipped': int(num_clipped),
        # NEW components
        'baseline_penalty': float(baseline_penalty),
        'action_utilization': float(action_utilization_bonus),
        'diversity_bonus': float(diversity_bonus),
        # Summary
        'num_active_units': int(num_active),
        'action_magnitude': float(action_magnitude),
    }

    return total_reward, reward_breakdown


# Print confirmation that improved version is loaded
print("="*80)
print("LOADED: IMPROVED env_helpers with policy collapse fixes")
print("="*80)
print("New components:")
print("  1. baseline_penalty: -50 for inaction")
print("  2. action_utilization_bonus: +5 * action_magnitude")
print("  3. diversity_bonus: +2 * num_active_units")
print("  4. flexibility stuck penalty: -20 if SoC=0.5 with action=0")
print("="*80)
