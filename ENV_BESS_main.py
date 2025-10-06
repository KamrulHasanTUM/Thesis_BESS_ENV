"""
ENV_BESS_main.py

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
import numpy as np

import env_helpers as helpers

warnings.simplefilter(action='ignore', category=FutureWarning)

class ENV_BESS(gymnasium.Env):
    """
    Battery Energy Storage System (BESS) environment for grid congestion management.
    Inherits from gymnasium.Env and implements step/reset methods for RL training.
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

                 # ========== BESS parameters (new) ==========
                 num_bess=5,
                 bess_capacity_mwh=50.0,
                 bess_power_mw=50.0,
                 soc_min=0.1,
                 soc_max=0.9,
                 initial_soc=0.5,
                 efficiency=0.9,
                 time_step_hours=1.0
                 ):
        """
        Initialize the BESS environment with network and training parameters.

        Grid Parameters (inherited from ENV_RHV):
            simbench_code: SimBench network identifier
            case_study: Load case ('bc' = base case)
            is_train: Training mode flag
            is_normalize: Data normalization flag
            max_step: Maximum steps per episode
            allowed_lines: Number of lines allowed to be disconnected
            convergence_penalty: Penalty for power flow non-convergence
            line_disconnect_penalty: Penalty for excessive line disconnections
            nan_vm_pu_penalty: Penalty for invalid voltage values
            penalty_scalar: Scaling factor for dynamic penalties
            bonus_constant: Reward weight for congestion relief
            exp_code: Experiment identifier

        BESS Parameters (new):
            num_bess: Number of BESS units (default: 5)
            bess_capacity_mwh: Energy capacity per unit in MWh (default: 50.0)
            bess_power_mw: Maximum charge/discharge power per unit in MW (default: 50.0)
            soc_min: Minimum allowed state of charge (default: 0.1 = 10%)
            soc_max: Maximum allowed state of charge (default: 0.9 = 90%)
            initial_soc: Starting SoC for all units (default: 0.5 = 50%)
            efficiency: Round-trip efficiency (default: 0.9 = 90%)
            time_step_hours: Duration of each timestep in hours (default: 1.0)
        """
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
        # These will be used by BESS-specific helper functions (initialize_bess_state, apply_bess_action, etc.)
        self.num_bess = num_bess
        self.bess_capacity_mwh = bess_capacity_mwh
        self.bess_power_mw = bess_power_mw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_soc = initial_soc
        self.efficiency = efficiency
        self.time_step_hours = time_step_hours

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
        """Reset the environment to initial state."""
        helpers.reset_episode_state(self)
        helpers.update_timestep_index(self, ts)
        helpers.reset_network_to_initial_state(self)
        
        helpers.initialize_bess_state(self)
        self.bess_sgen_indices = None
        
        validation_error = helpers.validate_grid_state_after_reset(self)
        if validation_error:
            return self.observation if hasattr(self, 'observation') else {}, self.info
        
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

        # Get max loading before action
        max_loading_before = self.net.res_line['loading_percent'].max()
        print(f"Max loading before: {max_loading_before}")

        # Apply BESS action: update sgen elements in network
        # - Converts continuous action array to power setpoints (MW)
        # - Updates pandapower sgen elements at BESS locations with commanded power
        # - Clips actions to [-bess_power_mw, +bess_power_mw] if needed
        # - Stores commanded action in env.bess_power for observation
        # - Power flow validation happens next to compute actual grid response
        helpers.apply_bess_action(self, action)

        # Validate grid state and run power flow
        # - Runs AC power flow with BESS injections/absorptions
        # - Checks for convergence errors, line disconnections, voltage violations
        # - Returns early with penalty if grid state is invalid
        # - Power flow results (res_sgen) contain actual BESS power after grid constraints
        error_result = helpers.validate_grid_state_after_action(self)
        if error_result:
            return error_result

        # Update BESS State of Charge based on actual power flow results
        # - Reads actual power from res_sgen (may differ from commanded action due to grid limits)
        # - Applies energy balance physics: ΔE = P × Δt with efficiency losses
        # - Updates env.bess_soc with new SoC values
        # - Clips SoC to [soc_min, soc_max] range (e.g., 10-90%)
        # - Updates env.bess_power with actual power for next observation
        # - MUST happen AFTER power flow (needs res_sgen results)
        helpers.update_bess_soc(self)

        # Get max loading after action
        max_loading_after = self.net.res_line['loading_percent'].max()
        print(f"Max loading after: {max_loading_after}")

        # Calculate BESS-aware reward for congestion management
        # - Primary objective: Congestion relief (reward = bonus × loading reduction)
        # - Secondary: SoC boundary penalties (prevent getting stuck at limits)
        # - Tertiary: Energy efficiency penalties (encourage smooth operation)
        # - Returns total reward and breakdown dict for logging
        reward, reward_breakdown = helpers.calculate_bess_reward(self, max_loading_before, max_loading_after)
        self.count += 1
        # Check episode completion
        if self.count >= self.max_step:
            self.truncated = True
            self.terminated = True
            return self.observation, reward, self.terminated, self.truncated, self.info

        # Update to next timestep
        # - Advances time_step index
        # - Resets network to initial topology
        # - Applies new load/generation profile from time series
        # - Runs power flow with updated conditions
        # - Builds new observation (includes updated grid state and BESS state)
        self.observation, self.terminated = helpers.update_to_next_timestep(self)
        
        print('count=', self.count)

        return self.observation, reward, self.terminated, self.truncated, self.info

    def get_data_length(self):
        """
        Return test and train data lengths.
        """
        return self.test_data_length, self.train_data_length



# Import configuration and training functions
from config import load_config, create_bess_env_config, create_training_config, save_training_metadata
from training import setup_environment, create_model, train_model, get_logdir
from utils import TQDMProgressCallback


def main():
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta) 
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
