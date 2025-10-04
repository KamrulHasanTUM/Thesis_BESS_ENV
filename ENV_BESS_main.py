"""
environment.py

Defines the ENV_BESS class — a custom Gymnasium-compatible reinforcement learning environment
for Battery Energy Storage System (BESS) based congestion management in high-voltage distribution grids.
The environment integrates SimBench benchmark networks with time-series load and generation profiles,
and performs AC load flow simulations using pandapower.

This environment is based on the ENV_RHV structure for topology reconfiguration, but adapted for
continuous BESS control actions. BESS-specific logic (state of charge management, power injection/absorption,
placement strategies) will be added incrementally.

Key features to be implemented:
- Continuous action space for BESS charging/discharging power control.
- Multi-modal observation space combining BESS state (SoC, location, capacity) with grid measurements
  (vm_pu, line loading, power injections).
- Congestion-oriented reward function accounting for BESS operational costs and grid relief.
- Episode termination conditions aligned with power flow convergence, voltage validity, and
  BESS operational constraints (SoC limits, power limits).

This environment supports reproducible experimentation with policy-based RL algorithms
for BESS-based grid congestion management under high loading scenarios.
"""

import gymnasium
import simbench as sb
import pandapower as pp
import warnings

import env_helpers as helpers

warnings.simplefilter(action='ignore', category=FutureWarning)

class ENV_BESS(gymnasium.Env):
    """
    Battery Energy Storage System (BESS) environment for grid congestion management.
    Inherits from gymnasium.Env and implements step/reset methods for RL training.
    """

    def __init__(self,
                 simbench_code="1-HV-mixed--0-sw",
                 case_study= 'bc',
                 is_train = True,
                 is_normalize = False,
                 max_step = 50,
                 allowed_lines = 200,
                 convergence_penalty = -200,
                 line_disconnect_penalty = -200,
                 nan_vm_pu_penalty = -200,
                 rho_min = 0.45,
                 action_type = 'NodeSplittingExEHVCBs',
                 exp_code = None,
                 penalty_scalar = -10,
                 bonus_constant = 10,
                 ):
        """
        Initialize the BESS environment with network and training parameters.
        """
        super().__init__()

        # Load network first
        self.net = self.load_simbench_network(simbench_code)

        # Initialize all parameters using helper functions
        helpers.initialize_config_parameters(self, simbench_code, case_study, is_train, is_normalize,
                                            max_step, action_type, exp_code, bonus_constant)
        helpers.initialize_penalty_parameters(self, allowed_lines, convergence_penalty, line_disconnect_penalty,
                                             nan_vm_pu_penalty, rho_min, penalty_scalar)
        helpers.initialize_state_variables(self)

        # TODO: Add BESS initialization here
        # Initialize BESS parameters: number of units, capacity (MWh), max power (MW),
        # initial SoC (%), placement strategy, efficiency, etc.

        # Load network and setup environment
        self.initial_net = self.setup_study_case(case_study, self.is_train, load_all=True)
        self.action_space, self.observation_space = self.create_action_and_observation_spaces()
        _ = self.reset()


    def load_simbench_network(self, simbench_code):
        """
        Load SimBench network from JSON file or SimBench repository.
        """
        if "nominal" in simbench_code:
            net = pp.from_json(f"{simbench_code}.json")
        else:
            net = sb.get_simbench_net(simbench_code)
        return net

    def setup_study_case(self, case_study, is_train, load_all=True):
        """
        Setup the study case and load time-series data for loads and generators.
        Prepares train/test splits for episodic training.
        """
        if load_all:
            print("Init")
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
        """
        Create action and observation spaces for the environment.
        """
        # TODO: Replace with BESS action/observation spaces
        # Action space: Continuous Box space for BESS power control (e.g., [-max_power, +max_power] per BESS)
        # Observation space: Add BESS state variables (SoC %, available capacity, current power, location info)
        # to existing grid observations (vm_pu, line loading, etc.)

        action_space = helpers.create_bess_action_space(self)  # TODO: Will be finalized in Step 4
        observation_space = helpers.create_bess_observation_space(self.net, self.num_bess, self.bess_power_mw)
        return action_space, observation_space


    def set_to_all_data(self):
        """
        Set data to full normalized dataset (for evaluation).
        """
        self.load_data = self.load_data_normalized
        self.sgen_data = self.sgen_data_normalized
        return


    def reset(self, options=None, seed=None, ts=None):
        """
        Reset the environment to initial state.
        Returns initial observation and info dict following Gymnasium API.
        """
        # Reset episode state variables
        helpers.reset_episode_state(self)
        helpers.update_timestep_index(self, ts)
        helpers.reset_network_to_initial_state(self)

        # TODO: Initialize BESS state (SoC, locations)
        # Reset BESS SoC to initial values (e.g., 50% or specified initial_soc)
        # Set BESS locations if dynamic placement is used
        # Reset BESS power output to 0

        # Validate grid state
        validation_error = helpers.validate_grid_state_after_reset(self)
        if validation_error:
            return self.observation, self.info

        # Build and return observation
        self.observation = helpers.build_observation_from_grid_state(self)
        return self.observation, self.info



    def step(self, action):
        """
        Execute one step in the environment.
        Applies BESS actions, runs power flow, calculates reward, and updates state.
        """
        if self.terminated:
            print("Reset Needed")
            return self.observation, 0, self.terminated, self.truncated, self.info

        # Apply action and check for errors
        max_loading_before = self.net.res_line['loading_percent'].max()
        print(f"Max loading before: {max_loading_before}")

        # TODO: Apply BESS actions before power flow
        # Convert action to BESS power setpoints (MW)
        # Update pandapower network: add/modify storage elements with P_mw values
        # Validate BESS constraints (SoC limits, power limits)

        # TODO: Apply BESS actions (will be implemented in Step 7)
        pass

        # Validate action results
        error_result = helpers.validate_grid_state_after_action(self)
        if error_result:
            return error_result

        # TODO: Update BESS SoC after power flow
        # Calculate energy change: delta_E = P_bess * time_step_hours
        # Update SoC: new_soc = old_soc + (delta_E / capacity) * 100
        # Clip SoC to valid range (0-100%)

        # Calculate reward
        max_loading_after = self.net.res_line['loading_percent'].max()
        print(f"Max loading after: {max_loading_after}")

        # TODO: Calculate BESS-aware reward
        # Include BESS operational costs (cycling, energy throughput)
        # Penalize SoC violations or deep discharge
        # Bonus for congestion relief while managing BESS efficiently

        reward = helpers.calculate_reward_for_step(self, max_loading_before, max_loading_after)

        # Check episode completion
        if self.count >= self.max_step:
            self.truncated = True
            self.terminated = True
            return self.observation, reward, self.terminated, self.truncated, self.info

        # Update to next state
        self.observation, self.terminated = helpers.update_to_next_timestep(self)
        self.count += 1
        print('count=', self.count)

        return self.observation, reward, self.terminated, self.truncated, self.info

    def get_data_length(self):
        """
        Return test and train data lengths.
        """
        return self.test_data_length, self.train_data_length



# Import configuration and training functions
from config import load_config, create_env_config, create_training_config, save_training_metadata
from training import setup_environment, create_model, train_model, get_logdir
from utils import TQDMProgressCallback


def main():
    """Main training pipeline."""
    # Load configurations
    init_meta = load_config()
    env_config = create_env_config(init_meta)
    training_config = create_training_config(init_meta)

    # Setup environment and logging
    env = setup_environment(ENV_BESS, env_config)
    logdir = get_logdir()

    # Save metadata and create model
    save_training_metadata(training_config, logdir)
    model = create_model(env, training_config, logdir)

    # Setup callbacks and train
    tqdm_callback = TQDMProgressCallback(total_timesteps=training_config['total_timesteps'])
    train_model(model, training_config, tqdm_callback)


if __name__ == "__main__":
    main()
