"""
env_helpers.py

Helper functions for the ENV_RHV reinforcement learning environment.
Contains functions for initialization, data preparation, observation/action space creation,
state management, validation, and reward calculation.
"""

import numpy as np
import copy
import json
import pandapower as pp
import simbench as sb
from sklearn.model_selection import train_test_split
from gymnasium.spaces import MultiDiscrete, Box
from gymnasium import spaces


# ==================== Initialization Helpers ====================

def initialize_config_parameters(env, simbench_code, case_study, is_train, is_normalize,
                                max_step, action_type, exp_code, bonus_constant):
    """Initialize configuration parameters for the environment."""
    env.simbench_code = simbench_code
    env.case_study = case_study
    env.is_train = is_train
    env.is_normalize = is_normalize
    env.max_step = max_step
    env.action_type = action_type
    env.exp_code = exp_code
    env.bonus_constant = bonus_constant


def initialize_penalty_parameters(env, allowed_lines, convergence_penalty, line_disconnect_penalty,
                                  nan_vm_pu_penalty, rho_min, penalty_scalar):
    """Initialize penalty and reward parameters."""
    env.allowed_lines = allowed_lines
    env.convergence_penalty = convergence_penalty
    env.line_disconnect_penalty = line_disconnect_penalty
    env.nan_vm_pu_penalty = nan_vm_pu_penalty
    env.rho_min = rho_min
    env.penalty_scalar = penalty_scalar

    # Error counters
    env.line_disconnect_count = 0
    env.convergence_error_count = 0
    env.nan_vm_pu_count = 0

    # Reward parameters
    env.gamma = 0.99  # Discount factor
    env.rho_max = 1.0  # Maximum acceptable load rate


def initialize_state_variables(env):
    """Initialize state and tracking variables."""
    env.initial_net = None
    env.relative_index = None
    env.time_step = -1
    env.observation = None

    # Data containers
    env.profiles = None
    env.gen_data = None
    env.load_data = None
    env.sgen_data = None

    # Episode state
    env.truncated = False
    env.terminated = False
    env.info = dict()
    env.count = 0

    # Data length tracking
    env.test_data_length = None
    env.train_data_length = None
    env.override_timestep = None


# ==================== Data Loading and Preparation ====================

def load_simbench_profiles_and_cases(net, case_study):
    """Load load cases and time-series profiles from SimBench."""
    loadcases = sb.get_absolute_values(net, profiles_instead_of_study_cases=False)
    net = apply_absolute_values_to_network(net, loadcases, case_study)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    return net, profiles


def prepare_train_test_split(profiles):
    """Prepare and split time-series data into train and test sets."""
    # Extract and normalize data
    load_data_raw = profiles[('load', 'p_mw')]
    sgen_data_raw = profiles[('sgen', 'p_mw')]
    load_data_normalized = load_data_raw.fillna(0)
    sgen_data_normalized = sgen_data_raw.fillna(0)

    # Split into train and test sets (80% train, 20% test)
    load_train, load_test = train_test_split(load_data_normalized, test_size=0.2, shuffle=False)
    sgen_train, sgen_test = train_test_split(sgen_data_normalized, test_size=0.2, shuffle=False)

    test_data_length = sgen_test.shape[0]
    train_data_length = sgen_train.shape[0]

    return {
        'load_data_normalized': load_data_normalized,
        'sgen_data_normalized': sgen_data_normalized,
        'load_train': load_train,
        'load_test': load_test,
        'sgen_train': sgen_train,
        'sgen_test': sgen_test,
        'test_data_length': test_data_length,
        'train_data_length': train_data_length
    }


def save_environment_metadata(env, data_splits):
    """Save environment metadata to JSON file."""
    env_meta = {
        "is_train": env.is_train,
        "is_normalize": env.is_normalize,
        "max_step": env.max_step,
        "case_study": env.case_study,
        "simbench_code": env.simbench_code,
        "allowed_lines": env.allowed_lines,
        "convergence_penalty": env.convergence_penalty,
        "line_disconnect_penalty": env.line_disconnect_penalty,
        "nan_vm_pu_penalty": env.nan_vm_pu_penalty,
        "rho_min": env.rho_min,
        "train_data_length": env.train_data_length,
        "test_data_length": env.test_data_length,
        "load_data_shape": data_splits['load_data_normalized'].shape,
        "sgen_data_shape": data_splits['sgen_data_normalized'].shape,
        "load_train_data_shape": data_splits['load_train'].shape,
        "sgen_train_data_shape": data_splits['sgen_train'].shape,
        "load_test_data_shape": data_splits['load_test'].shape,
        "sgen_test_data_shape": data_splits['sgen_test'].shape,
        "exp_code": env.exp_code,
        "action_type": env.action_type,
        "penalty_scalar": env.penalty_scalar,
        "total_CBs": env.net.switch[(env.net.switch['et'] == 'b') & (env.net.switch['type'] == 'CB')].shape[0],
        "total_switches": env.net.switch.shape[0],
        "bonus_constant": env.bonus_constant,
    }

    with open("env_meta.json", "w") as file:
        json.dump(env_meta, file, indent=4)
    print("Data saved to env_meta.json")
    print(env_meta)

    return env_meta


def apply_absolute_values_to_network(net, absolute_values_dict, case_or_time_step):
    """Apply absolute values from profiles to network elements."""
    for elm_param in absolute_values_dict.keys():
        if absolute_values_dict[elm_param].shape[1]:
            elm = elm_param[0]
            param = elm_param[1]
            net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]
    return net


# ==================== Action and Observation Space Creation ====================

def create_bess_action_space(env):
    """
    Create continuous action space for BESS power dispatch control.

    This function defines a Box (continuous) action space where each dimension corresponds
    to the power setpoint (MW) for one BESS unit. The action space enables fine-grained
    control of battery charging/discharging for optimal congestion management.

    Action Space Structure:
    - Type: Box (continuous)
    - Shape: (num_bess,) - one action per BESS unit
    - Range: [-bess_power_mw, +bess_power_mw] per unit
    - dtype: np.float32

    Sign Convention:
    - Negative values: Charging (energy flows INTO battery from grid)
    - Positive values: Discharging (energy flows OUT OF battery to grid)
    - Zero: Idle/standby

    Physical Interpretation (Example: 5 BESS units, 50 MW rating, 50 MWh capacity):
    - bess_power_mw = 50 MW represents 1C-rate (1-hour discharge/charge time)
    - Action bounds ±50 MW allow full utilization of battery power capability
    - 50 MW × 1 hour = 50 MWh (matches capacity for complete charge/discharge cycle)

    Example Action:
        action = np.array([-50.0, 0.0, 50.0, 30.0, -40.0])

        Interpretation:
        - BESS 0: Charging at maximum power (50 MW) - absorbing excess generation
        - BESS 1: Idle (0 MW) - standby mode
        - BESS 2: Discharging at maximum power (50 MW) - supporting grid during peak load
        - BESS 3: Discharging at 30 MW (60% of capacity) - partial grid support
        - BESS 4: Charging at 40 MW (80% of capacity) - absorbing moderate excess generation

    Args:
        env: Environment instance with BESS configuration parameters:
            - env.num_bess: Number of BESS units
            - env.bess_power_mw: Maximum charge/discharge power per unit (MW)

    Returns:
        spaces.Box: Continuous action space for BESS power control

    Note:
        - Continuous Box space is essential for optimal dispatch (vs. discrete actions)
        - Enables gradient-based RL algorithms (PPO, SAC, DDPG) to learn smooth control policies
        - Physical power limits (±bess_power_mw) correspond to inverter rating constraints
        - SoC constraints will be enforced separately during step() execution
    """
    # Use Box space for continuous power control
    # This allows fine-grained dispatch optimization (e.g., action = 23.7 MW, not just discrete levels)
    # Essential for congestion management where partial power adjustments are often needed
    action_space = Box(
        low=-env.bess_power_mw,      # Maximum charging power (negative = charging)
        high=env.bess_power_mw,      # Maximum discharging power (positive = discharging)
        shape=(env.num_bess,),       # One power setpoint per BESS unit
        dtype=np.float32             # Float32 for RL framework compatibility
    )

    # Physical meaning of bounds:
    # - For 50 MW rating: ±50 MW corresponds to 1C-rate (1-hour full charge/discharge)
    # - Lower C-rates (e.g., 25 MW = 0.5C) extend battery lifetime
    # - Higher utilization provides more grid flexibility but increases degradation
    # - Continuous actions enable the agent to learn optimal power-lifetime tradeoffs

    return action_space


def create_bess_observation_space(net, num_bess, bess_power_mw):
    """
    Create observation space combining grid state and BESS state information.

    This function extends the standard grid observation space by adding BESS-specific
    state variables that enable the RL agent to make informed dispatch decisions
    considering both grid conditions and battery status.

    Observation Space Structure:
    - Type: Dict (multi-modal observation space)
    - Components: 7 grid observations + 2 BESS observations

    Grid Observations (from ENV_RHV):
    1. discrete_switches: Switch states and line status (binary)
       - Provides network topology information
       - Helps agent understand current grid configuration

    2. continuous_vm_bus: Bus voltages in per-unit (0.5-1.5 p.u.)
       - Critical for voltage stability monitoring
       - Agent learns to prevent over/under-voltage violations

    3. continuous_sgen_data: Generator power output (0-100000 MW)
       - Renewable generation profile (solar, wind)
       - Agent adapts BESS dispatch to generation patterns

    4. continuous_load_data: Load power consumption (0-100000 MW)
       - Demand profile
       - Agent learns peak shaving and valley filling strategies

    5. continuous_line_loadings: Line loading percentages (0-800%)
       - Primary congestion indicator
       - Agent optimizes BESS to reduce overloading

    6. continuous_space_ext_grid_p_mw: External grid active power (±50M MW)
       - Grid import/export
       - Agent minimizes grid dependency using BESS

    7. continuous_space_ext_grid_q_mvar: External grid reactive power (±50M MVAr)
       - Reactive power flow
       - Supports voltage control decisions

    BESS Observations (new):
    8. bess_soc: State of Charge for each BESS (0.0-1.0, normalized)
       - 0.0 = completely empty (0% SoC)
       - 1.0 = completely full (100% SoC)
       - Normalized for RL algorithm stability (avoids large value ranges)
       - Enables agent to track available energy capacity
       - Critical for planning multi-step discharge/charge sequences

    9. bess_power: Current power output for each BESS (±bess_power_mw)
       - Negative values: charging (e.g., -50 MW)
       - Positive values: discharging (e.g., +50 MW)
       - Zero: idle/standby
       - Agent observes current dispatch state for temporal consistency
       - Helps learn smooth power ramp rates (avoid sudden changes)

    Example Observation (5 BESS units, 50 MW rating):
        observation = {
            'discrete_switches': np.array([1, 0, 1, 1, 0, ...]),  # Binary topology
            'continuous_vm_bus': np.array([1.02, 0.98, 1.00, ...]),  # Voltages (p.u.)
            'continuous_sgen_data': np.array([120.5, 80.3, ...]),  # Gen power (MW)
            'continuous_load_data': np.array([200.0, 150.0, ...]),  # Load power (MW)
            'continuous_line_loadings': np.array([85.2, 120.5, ...]),  # Loading (%)
            'continuous_space_ext_grid_p_mw': np.array([500.0]),  # Grid import (MW)
            'continuous_space_ext_grid_q_mvar': np.array([50.0]),  # Grid Q (MVAr)

            # BESS observations
            'bess_soc': np.array([0.5, 0.3, 0.8, 0.6, 0.4]),  # SoC: 50%, 30%, 80%, 60%, 40%
            'bess_power': np.array([-20.0, 0.0, 50.0, 30.0, -40.0])  # MW dispatch
        }

        Interpretation:
        - BESS 0: 50% SoC, charging at 20 MW (building reserves)
        - BESS 1: 30% SoC, idle (low energy, ready to charge)
        - BESS 2: 80% SoC, discharging at max 50 MW (peak support)
        - BESS 3: 60% SoC, discharging at 30 MW (partial support)
        - BESS 4: 40% SoC, charging at 40 MW (storing excess generation)

    Why These Observations Matter for RL:
    - SoC tracking: Agent learns when BESS can charge/discharge without violating limits
    - Power observability: Agent maintains awareness of current dispatch for smooth control
    - Grid-BESS coupling: Agent correlates grid stress (line loading) with BESS availability
    - Multi-step planning: SoC enables lookahead (e.g., reserve capacity for predicted peak)
    - Constraint learning: Agent implicitly learns operational bounds from observation ranges

    Args:
        net: Pandapower network object with grid topology
        num_bess: Number of BESS units in the system
        bess_power_mw: Maximum power rating per BESS unit (MW)

    Returns:
        spaces.Dict: Combined observation space with grid and BESS state
    """
    # Get grid dimensions (same as create_observation_space)
    num_lines = net.line.shape[0]
    num_bus = net.bus.shape[0]
    num_sgenerators = net.sgen.shape[0]
    num_loads = net.load.shape[0]
    num_ext_grid = net.ext_grid.shape[0]

    # Discrete space: switches and line status (preserved from ENV_RHV)
    discrete_space = MultiDiscrete([2] * net.switch.shape[0] + [2] * num_lines)

    # Define combined observation space
    observation_spaces = {
        # ========== Grid Observations (from ENV_RHV) ==========
        "discrete_switches": discrete_space,
        "continuous_vm_bus": Box(low=0.5, high=1.5, shape=(num_bus,), dtype=np.float32),
        "continuous_sgen_data": Box(low=0.0, high=100000, shape=(num_sgenerators,), dtype=np.float32),
        "continuous_load_data": Box(low=0.0, high=100000, shape=(num_loads,), dtype=np.float32),
        "continuous_line_loadings": Box(low=0.0, high=800.0, shape=(num_lines,), dtype=np.float32),
        "continuous_space_ext_grid_p_mw": Box(low=-50000000, high=50000000, shape=(num_ext_grid,), dtype=np.float32),
        "continuous_space_ext_grid_q_mvar": Box(low=-50000000, high=50000000, shape=(num_ext_grid,), dtype=np.float32),

        # ========== BESS Observations (new) ==========
        # State of Charge (SoC) - normalized to [0, 1] range
        # - Normalization benefits: (1) RL algorithms prefer inputs in similar ranges (0-1)
        #   (2) avoids large gradients, (3) easier to interpret (0=empty, 1=full)
        # - Agent uses SoC to decide: Can I discharge more? Should I stop charging?
        # - Enables multi-step reasoning: "Save 20% SoC for evening peak"
        "bess_soc": Box(
            low=0.0,              # 0% SoC (completely empty, but actual min is soc_min=10%)
            high=1.0,             # 100% SoC (completely full, but actual max is soc_max=90%)
            shape=(num_bess,),    # One SoC value per BESS unit
            dtype=np.float32
        ),

        # Current power output - same range as action space for consistency
        # - Agent observes previous dispatch decision effects
        # - Enables learning of smooth control (avoid power jumps that stress inverters)
        # - Temporal correlation: Agent can detect if actions are being executed correctly
        # - Example: If agent commanded -50 MW but observes -30 MW, it learns grid constraints
        "bess_power": Box(
            low=-bess_power_mw,   # Maximum charging power (negative convention)
            high=bess_power_mw,   # Maximum discharging power (positive convention)
            shape=(num_bess,),    # One power value per BESS unit
            dtype=np.float32
        ),
    }

    return spaces.Dict(observation_spaces)


# ==================== Reset Helpers ====================

def reset_episode_state(env):
    """Reset episode tracking variables."""
    env.terminated = False
    env.truncated = False
    env.count = 0


def update_timestep_index(env, ts):
    """Update timestep for the episode."""
    if ts is None:
        env.time_step += 1
        if env.time_step >= env.sgen_data.shape[0]:
            env.time_step = 0
        env.relative_index = env.sgen_data.index.values[env.time_step]
    else:
        env.time_step = ts
        env.relative_index = env.sgen_data.index.values[env.time_step]


def reset_network_to_initial_state(env):
    """Reset network to initial state and apply current timestep values."""
    env.net = copy.deepcopy(env.initial_net)
    env.net = apply_absolute_values_to_network(env.net, env.profiles, env.relative_index)

    # Run load flow calculations
    try:
        pp.runpp(env.net)
    except:
        env.terminated = True
        env.truncated = True
        print("Load flow error in resetting")
        env.convergence_error_count += 1


def validate_grid_state_after_reset(env):
    """Validate grid state and return True if invalid."""
    if env.terminated:
        return True

    if env.net.res_line['loading_percent'].isna().sum() > env.allowed_lines:
        env.terminated = True
        env.truncated = True
        print("Line disconnect error in resetting")
        env.line_disconnect_count += 1
        return True

    if env.net.res_bus['vm_pu'].isna().any():
        env.terminated = True
        env.truncated = True
        print("Vm pu error in resetting")
        env.nan_vm_pu_count += 1
        return True

    return False


def build_observation_from_grid_state(env):
    """Build observation dictionary from current grid state."""
    loading_percent = env.net.res_line['loading_percent'].fillna(0).values.astype(np.float32)
    vm_pu = env.net.res_bus['vm_pu'].fillna(0).values.astype(np.float32)

    discrete_switches = np.concatenate([
        env.net.switch['closed'].astype(int).values,
        env.net.line['in_service'].astype(int).values
    ])

    return {
        "discrete_switches": discrete_switches,
        "continuous_vm_bus": vm_pu,
        "continuous_sgen_data": env.sgen_data.values[env.time_step].astype(np.float32),
        "continuous_load_data": env.load_data.values[env.time_step].astype(np.float32),
        "continuous_line_loadings": loading_percent,
        "continuous_space_ext_grid_p_mw": env.net.res_ext_grid['p_mw'].fillna(0).values.astype(np.float32),
        "continuous_space_ext_grid_q_mvar": env.net.res_ext_grid['q_mvar'].fillna(0).values.astype(np.float32)
    }


# ==================== Step Helpers ====================

def validate_grid_state_after_action(env):
    """Validate grid state after action and return error result if invalid."""
    # Run load flow calculations
    try:
        pp.runpp(env.net)
        print("Load flow passed in stepping")
    except:
        env.terminated = True
        env.truncated = True
        print("Load flow error in stepping")
        env.convergence_error_count += 1
        return env.observation, env.convergence_penalty, env.terminated, env.truncated, env.info

    # Check line disconnections
    if env.net.res_line['loading_percent'].isna().sum() > env.allowed_lines:
        env.terminated = True
        print("Line disconnect error")
        env.line_disconnect_count += 1
        return env.observation, env.line_disconnect_penalty, env.terminated, env.truncated, env.info

    # Check voltage violations
    if env.net.res_bus['vm_pu'].isna().any():
        env.terminated = True
        print("Vm pu error")
        env.nan_vm_pu_count += 1
        penalty = calculate_voltage_violation_penalty(env)
        return env.observation, penalty, env.terminated, env.truncated, env.info

    print("ALL WORKING")
    return None


def calculate_voltage_violation_penalty(env):
    """Calculate penalty for voltage violations."""
    if env.nan_vm_pu_penalty == "dynamic":
        penalty = env.penalty_scalar * env.net.res_bus['vm_pu'].isna().sum()
        print(f"Penalty: {penalty}")
        return penalty
    else:
        return env.nan_vm_pu_penalty


def calculate_reward_for_step(env, max_loading_before, max_loading_after):
    """Calculate reward for the current step."""
    env.net.res_line['loading_percent'] = env.net.res_line['loading_percent'].fillna(0)
    P_j_t = np.array([line['loading_percent'] / 100 for _, line in env.net.res_line.iterrows()])
    return calculate_congestion_reward(env, P_j_t, max_loading_before, max_loading_after)


def calculate_congestion_reward(env, rho, max_loading_before, max_loading_after):
    """Calculate congestion-based reward with bonus for reducing loading."""
    _temp = np.zeros(len(rho))
    for i in range(len(rho)):
        _temp[i] = np.max([env.rho_min, rho[i]])
    u_t = np.sum(1 - (_temp - env.rho_min))
    print(f"Congestion: {u_t}")

    bonus = env.bonus_constant * (max_loading_before - max_loading_after)
    print(f"Bonus: {bonus}")

    R_congestion = u_t + bonus
    print(f"R_congestion: {R_congestion}")
    return R_congestion


# ==================== Update State Helpers ====================

def update_to_next_timestep(env):
    """Update environment to the next timestep."""
    env.time_step += 1
    if env.time_step >= env.sgen_data.shape[0]:
        env.time_step = 0

    env.net = copy.deepcopy(env.initial_net)
    env.relative_index = env.sgen_data.index.values[env.time_step]
    env.net = apply_absolute_values_to_network(env.net, env.profiles, env.relative_index)

    # Run load flow calculations
    try:
        pp.runpp(env.net)
        print("Load flow passed in updating")
    except:
        print("Load flow error in updating")
        env.convergence_error_count += 1
        return env.observation, True

    # Validate new state
    loading_percent = env.net.res_line['loading_percent']
    if loading_percent.isna().sum() > env.allowed_lines:
        print("Line disconnect error in updating")
        env.line_disconnect_count += 1
        return env.observation, True

    if env.net.res_bus['vm_pu'].isna().any():
        print("Vm pu error in updating")
        env.nan_vm_pu_count += 1
        return env.observation, True

    # Build new observation
    observation = build_observation_from_grid_state(env)
    return observation, False
