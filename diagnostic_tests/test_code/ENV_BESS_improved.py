"""
IMPROVED VERSION - ENV_BESS with fixes for policy collapse

This is a TESTING version that includes minimal changes to support improved reward calculation.

CHANGES FROM ORIGINAL:
1. Stores last action so reward function can access it
2. Uses improved env_helpers_improved instead of env_helpers
3. All other functionality remains IDENTICAL

DO NOT USE THIS IN PRODUCTION - FOR TESTING ONLY
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import original dependencies
import gymnasium
import simbench as sb
import pandapower as pp
import warnings
import numpy as np

# Import improved helpers from this folder
import env_helpers_improved as helpers

# Import config and training from parent
from config import load_config, create_bess_env_config, create_training_config, save_training_metadata
from training import setup_environment, create_model, get_logdir, train_model_with_wandb

warnings.simplefilter(action='ignore', category=FutureWarning)


class ENV_BESS(gymnasium.Env):
    """
    IMPROVED Battery Energy Storage System (BESS) environment.

    This version includes fixes for policy collapse by:
    - Storing last action for reward calculation
    - Using improved reward function
    - All other functionality unchanged
    """

    def __init__(self,
             # ========== Grid parameters (from ENV_RHV) ==========
             simbench_code="1-HV-mixed--0-sw",
             case_study='bc',
             is_train=True,
             is_normalize=False,
             max_step=50,
             allowed_lines=200,
             convergence_penalty=-200,
             line_disconnect_penalty=-200,
             nan_vm_pu_penalty=-200,
             penalty_scalar=-10,
             bonus_constant=10,
             exp_code=None,

             # ========== BESS parameters ==========
             num_bess=5,
             bess_capacity_mwh=50.0,
             bess_power_mw=50.0,
             soc_min=0.1,
             soc_max=0.9,
             initial_soc=0.5,
             efficiency=0.9,
             time_step_hours=0.25,
             bess_locations=None,

             # ========== Engineering constants ==========
             voltage_min_pu=0.5,
             voltage_max_pu=1.5,
             soc_boundary_margin=0.05,
             soc_penalty_weight=-0.2,
             **extra_kwargs
             ):
        """Initialize the IMPROVED BESS environment."""
        super().__init__()

        # Load network first
        self.net = self.load_simbench_network(simbench_code)

        # Initialize grid parameters using helper functions
        helpers.initialize_config_parameters(self, simbench_code, case_study, is_train, is_normalize,
                                            max_step, 'BESS', exp_code, bonus_constant)
        helpers.initialize_penalty_parameters(self, allowed_lines, convergence_penalty, line_disconnect_penalty,
                                             nan_vm_pu_penalty, 0.0, penalty_scalar)
        helpers.initialize_state_variables(self)

        # Initialize BESS parameters
        self.num_bess = num_bess
        self.bess_capacity_mwh = bess_capacity_mwh
        self.bess_power_mw = bess_power_mw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_soc = initial_soc
        self.efficiency = efficiency
        self.time_step_hours = time_step_hours

        # Engineering constants
        self.soc_boundary_margin = soc_boundary_margin
        self.voltage_min_pu = voltage_min_pu
        self.voltage_max_pu = voltage_max_pu
        self.soc_penalty_weight = soc_penalty_weight
        self.soc_clipping_penalty = extra_kwargs.pop("soc_clipping_penalty", -10.0)

        # Handle optional flexibility parameters
        flexibility_bonus_weight = extra_kwargs.pop("flexibility_bonus_weight", 0.0)
        soc_flex_center = extra_kwargs.pop("soc_flex_center", 0.5)
        soc_flex_width = extra_kwargs.pop("soc_flex_width", 0.2)

        self.flexibility_bonus_weight = flexibility_bonus_weight
        self.soc_flex_center = soc_flex_center
        self.soc_flex_width = soc_flex_width

        if extra_kwargs:
            unexpected = ", ".join(sorted(extra_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        # Store BESS locations
        self.bess_locations = bess_locations

        # NEW: Initialize last action storage for improved reward calculation
        self._last_action = np.zeros(num_bess)

        # Load network and setup environment
        self.initial_net = self.setup_study_case(case_study, self.is_train, load_all=True)
        self.action_space, self.observation_space = self.create_action_and_observation_spaces()
        _ = self.reset()

        print("="*80)
        print("LOADED: IMPROVED ENV_BESS with policy collapse prevention")
        print("="*80)

    def load_simbench_network(self, simbench_code):
        """Load SimBench network from JSON file or SimBench repository."""
        if "nominal" in simbench_code:
            net = pp.from_json(f"{simbench_code}.json")
        else:
            net = sb.get_simbench_net(simbench_code)
        return net

    def setup_study_case(self, case_study, is_train, load_all=True):
        """Setup the study case and load time-series data."""
        if load_all:
            print("Init IMPROVED ENV")
            self.case_study = case_study
            net, self.profiles = helpers.load_simbench_profiles_and_cases(self.net, case_study)

            data_splits = helpers.prepare_train_test_split(self.profiles)

            # Store normalized data and lengths
            self.load_data_normalized = data_splits['load_data_normalized']
            self.sgen_data_normalized = data_splits['sgen_data_normalized']
            self.test_data_length = data_splits['test_data_length']
            self.train_data_length = data_splits['train_data_length']

            if is_train:
                self.env_meta = helpers.save_environment_metadata(self, data_splits)
                self.load_data = data_splits['load_train']
                self.sgen_data = data_splits['sgen_train']
            else:
                self.load_data = data_splits['load_test']
                self.sgen_data = data_splits['sgen_test']

        return net

    def create_action_and_observation_spaces(self):
        """Create action and observation spaces."""
        action_space = helpers.create_bess_action_space(self)
        observation_space = helpers.create_bess_observation_space(self.net, self.num_bess, self.bess_power_mw)
        return action_space, observation_space

    def set_to_all_data(self):
        """Set data to full normalized dataset (for evaluation)."""
        self.load_data = self.load_data_normalized
        self.sgen_data = self.sgen_data_normalized
        return

    def reset(self, options=None, seed=None, ts=None, _retry_count=0):
        """Reset the environment state with automatic retries on convergence failure."""

        # Step 1: Reset episode counters and flags
        helpers.reset_episode_state(self)

        # Step 2: Select timestep (random or specified)
        helpers.update_timestep_index(self, ts)

        # Step 3: Reset network and run power flow
        helpers.reset_network_to_initial_state(self)

        # ========== CONVERGENCE FAILURE HANDLING (HYBRID APPROACH) ==========
        if self.terminated:
            # Power flow failed to converge at this timestep
            MAX_RETRIES = 10
            if _retry_count >= MAX_RETRIES:
                # After MAX_RETRIES failures, return safe default observation
                print(f"❌ CRITICAL: Failed to find valid timestep after {MAX_RETRIES} retries!")
                print("   This indicates a serious network/data problem.")
                print("   Returning zero-filled observation as fallback.")
                print("   Training may not work correctly - investigate network convergence.")

                default_observation = {}
                for key, space in self.observation_space.spaces.items():
                    default_observation[key] = np.zeros(space.shape, dtype=space.dtype)

                self.observation = default_observation
                return self.observation, self.info

            # Otherwise, retry with a new random timestep
            print(f"   Retry attempt {_retry_count + 1}/{MAX_RETRIES} - sampling new timestep...")
            return self.reset(options=options, seed=seed, ts=None, _retry_count=_retry_count + 1)
        # ========== END CONVERGENCE FAILURE HANDLING ==========

        # Step 4: Initialize BESS state (SoC, power, locations)
        helpers.initialize_bess_state(self, bess_locations=self.bess_locations)
        self.bess_sgen_indices = None

        # NEW: Reset last action for improved reward calculation
        self._last_action = np.zeros(self.num_bess)

        # Step 5: Additional validation checks (voltage, line disconnections)
        validation_error = helpers.validate_grid_state_after_reset(self)
        if validation_error:
            print(f"⚠️ Post-convergence validation failed (retry {_retry_count + 1}/10)")
            return self.reset(options=options, seed=seed, ts=None, _retry_count=_retry_count + 1)

        # Step 6: Build observation from valid grid state
        self.observation = helpers.build_observation_from_grid_state(self)

        # Step 7: Return observation and info
        return self.observation, self.info

    def step(self, action):
        """Execute one environment step using a BESS power setpoint vector."""

        # NEW: Store action for improved reward calculation
        self._last_action = action.copy()

        if self.terminated:
            print("Reset Needed")
            return self.observation, 0, self.terminated, self.truncated, self.info

        # 1. Record max loading before action
        max_loading_before = self.net.res_line['loading_percent'].max()
        self.loading_before_action = max_loading_before
        print(f"Max loading before: {max_loading_before}")

        # 2. Clip action against SoC/energy limits
        clipped_action_mw, clip_info = helpers.apply_bess_action_with_soc_clipping(self, action)
        self.info['clip_info'] = clip_info

        # 3. Apply the feasible action to the grid (already in MW)
        helpers.apply_bess_action(self, clipped_action_mw, already_scaled=True)

        # 4. Run power flow to validate the new state
        error_result = helpers.validate_grid_state_after_action(self)
        if error_result:
            return error_result

        # 5. Record max loading after action
        max_loading_after = self.net.res_line['loading_percent'].max()
        self.loading_after_action = max_loading_after
        print(f"Max loading after: {max_loading_after}")

        # 6. Update SoC and capture any clipping penalty
        helpers.update_bess_soc(self)

        # 7. Compute reward components using IMPROVED function
        reward, reward_breakdown = helpers.calculate_bess_reward(
            self, max_loading_before, max_loading_after
        )

        self.info['reward_breakdown'] = reward_breakdown
        self.info['bess_soc'] = self.bess_soc.copy()
        num_bounds = reward_breakdown.get('num_near_bounds', '?')
        print(f"SoC: {np.round(self.bess_soc, 3)}, Bounds: {num_bounds}, Reward Parts: {reward_breakdown}")

        # 8. Episode bookkeeping
        self.count += 1
        if self.count >= self.max_step:
            self.truncated = True
            self.terminated = True
            return self.observation, reward, self.terminated, self.truncated, self.info

        # 9. Update to next timestep
        self.observation, error_in_update = helpers.update_to_next_timestep(self)
        if error_in_update:
            self.terminated = True
            return self.observation, self.convergence_penalty, self.terminated, self.truncated, self.info

        # 10. Return step results
        return self.observation, reward, self.terminated, self.truncated, self.info


# Test import
if __name__ == "__main__":
    print("Testing IMPROVED ENV_BESS import...")
    print("SUCCESS: All imports working")
    print("\nThis environment includes:")
    print("  1. Inaction penalty (-50)")
    print("  2. Action utilization bonus (+5 * magnitude)")
    print("  3. Diversity bonus (+2 * active_units)")
    print("  4. Stuck state penalty (-20)")
