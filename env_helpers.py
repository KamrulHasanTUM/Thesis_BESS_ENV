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

def run_power_flow(env, context=""):
    """
    Execute power flow calculation with standardized error handling.
    
    Args:
        env: Environment instance
        context: String describing where this is called (for debugging)
    
    Returns:
        bool: True if converged, False otherwise
    """
    try:
        pp.runpp(env.net)
        if context:
            print(f"Load flow passed in {context}")
        return env.net.converged
    except Exception as e:
        if context:
            print(f"Load flow error in {context}: {e}")
        return False

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


def initialize_bess_state(env, bess_locations=None):
    """
    Initialize BESS state variables including SoC, power, and grid connection points.

    This function sets up the initial state for all BESS units in the environment,
    including their State of Charge (SoC), power output, and physical locations
    (bus connections) in the power grid. The initialization ensures BESS units
    start in a neutral, ready-to-dispatch state.

    BESS State Initialization:
    - bess_soc: State of Charge for each unit (0.0-1.0, normalized)
      * Initialized to env.initial_soc (typically 0.5 = 50%)
      * 50% provides bidirectional flexibility: can charge OR discharge
      * Represents a "ready" state with reserves for both peak shaving and valley filling

    - bess_power: Current power dispatch for each unit (MW)
      * Initialized to 0.0 (all units idle/standby at episode start)
      * Agent will determine first dispatch action based on grid conditions
      * Zero output prevents initial grid disturbance

    - bess_locations: Bus indices where BESS units are physically connected
      * If provided: Uses specified bus indices
      * If None: Automatically selects from 110kV buses
      * 110kV level is appropriate for grid-scale BESS (subtransmission voltage)

    Automatic Bus Selection (when bess_locations=None):
    - Filters network to find all 110kV buses (vn_kv == 110)
    - Randomly selects num_bess buses from this filtered set
    - Uses env.np_random if available (for reproducibility), else numpy.random
    - Rationale for 110kV:
      * Standard voltage for distribution grid BESS connections
      * High enough for MW-scale power injection without voltage issues
      * Low enough to directly impact congestion in HV distribution networks
      * Typical SimBench HV network voltage level

    Validation:
    - Verifies num_bess matches length of provided bess_locations
    - Checks all bus indices exist in the network
    - Raises ValueError with descriptive message if validation fails

    Parameters:
        env: Environment instance with BESS configuration:
            - env.num_bess: Number of BESS units
            - env.initial_soc: Starting SoC (e.g., 0.5 for 50%)
            - env.net: Pandapower network object
            - env.np_random (optional): NumPy random generator for reproducibility

        bess_locations (array-like, optional): Bus indices for BESS placement
            - If None: Auto-select from 110kV buses
            - If provided: Must be valid bus indices, length must equal num_bess

    Raises:
        ValueError: If validation fails (length mismatch, invalid bus indices)

    Example Usage:
        # Automatic bus selection (recommended for initial experiments)
        initialize_bess_state(env)
        # Result: env.bess_locations = [5, 12, 18, 25, 30] (random 110kV buses)
        #         env.bess_soc = [0.5, 0.5, 0.5, 0.5, 0.5] (50% SoC)
        #         env.bess_power = [0.0, 0.0, 0.0, 0.0, 0.0] (idle)

        # Manual bus specification (for targeted placement studies)
        initialize_bess_state(env, bess_locations=np.array([5, 12, 18, 25, 30]))
        # Result: BESS placed at specified buses, same SoC/power initialization

        # Access initialized state
        print(f"BESS at buses: {env.bess_locations}")
        print(f"Initial SoC: {env.bess_soc}")
        print(f"Initial power: {env.bess_power}")
    """
    # Initialize State of Charge (SoC) array
    # - Start at initial_soc (typically 50%) for bidirectional flexibility
    # - Agent can charge (absorb excess generation) OR discharge (support peak demand)
    # - Normalized to [0, 1] range for RL algorithm stability
    env.bess_soc = np.full(env.num_bess, env.initial_soc, dtype=np.float32)

    # Initialize power dispatch array
    # - Start at 0.0 MW (all BESS units idle)
    # - Prevents initial grid disturbance from arbitrary power injection
    # - Agent determines first action based on observed grid conditions
    env.bess_power = np.zeros(env.num_bess, dtype=np.float32)

    # Initialize BESS grid connection locations
    if bess_locations is not None:
        # Manual bus specification: validate provided locations
        bess_locations = np.array(bess_locations, dtype=np.int32)

        # Validation 1: Check length matches number of BESS units
        if len(bess_locations) != env.num_bess:
            raise ValueError(
                f"Length of bess_locations ({len(bess_locations)}) must match "
                f"num_bess ({env.num_bess})"
            )

        # Validation 2: Check all bus indices exist in network
        valid_bus_indices = set(env.net.bus.index)
        invalid_buses = [bus for bus in bess_locations if bus not in valid_bus_indices]
        if invalid_buses:
            raise ValueError(
                f"Invalid bus indices in bess_locations: {invalid_buses}. "
                f"Valid bus indices are: {sorted(valid_bus_indices)}"
            )

        env.bess_locations = bess_locations

    else:
        # Automatic bus selection: choose from 110kV buses
        # 110kV is the standard HV distribution voltage level for grid-scale BESS
        # - High enough for MW-scale injection without voltage regulation issues
        # - Low enough to directly mitigate congestion in distribution networks
        # - Typical voltage level in SimBench HV networks
        buses_110kv = env.net.bus[env.net.bus['vn_kv'] == 110].index.values

        if len(buses_110kv) < env.num_bess:
            raise ValueError(
                f"Not enough 110kV buses ({len(buses_110kv)}) to place "
                f"{env.num_bess} BESS units. Consider reducing num_bess or "
                f"manually specifying bess_locations with other voltage levels."
            )

        # Use environment's random generator if available (for reproducibility)
        # Otherwise fall back to numpy.random
        if hasattr(env, 'np_random') and env.np_random is not None:
            selected_buses = env.np_random.choice(buses_110kv, size=env.num_bess, replace=False)
        else:
            selected_buses = np.random.choice(buses_110kv, size=env.num_bess, replace=False)

        env.bess_locations = selected_buses.astype(np.int32)

    print(f"BESS initialized: {env.num_bess} units at buses {env.bess_locations}")
    print(f"Initial SoC: {env.bess_soc * 100}% | Initial power: {env.bess_power} MW")


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


def create_bess_observation_space(net, num_bess, bess_power_mw, voltage_min_pu=0.5, voltage_max_pu=1.5):
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
        "continuous_vm_bus": Box(low=voltage_min_pu, high=voltage_max_pu, shape=(num_bus,), dtype=np.float32),
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
    if not run_power_flow(env, "reset"):
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
    """
    Build observation dictionary from current grid state.

    This function constructs the observation space for the RL agent by extracting
    relevant grid state variables from the power flow results and time series data.
    For BESS environments, it also includes battery state information.

    Grid Observations (always included):
    - discrete_switches: Switch states and line status
    - continuous_vm_bus: Bus voltages (per-unit)
    - continuous_sgen_data: Generator power from time series
    - continuous_load_data: Load power from time series
    - continuous_line_loadings: Line loading percentages
    - continuous_space_ext_grid_p_mw: External grid active power
    - continuous_space_ext_grid_q_mvar: External grid reactive power

    BESS Observations (included only if BESS exists):
    - bess_soc: State of charge for each BESS unit (0-1, normalized)
    - bess_power: Current power output for each BESS unit (MW)

    Args:
        env: Environment instance with grid state and optional BESS state

    Returns:
        dict: Observation dictionary with grid state (and BESS state if applicable)
    """
    # Extract grid state from power flow results
    loading_percent = env.net.res_line['loading_percent'].fillna(0).values.astype(np.float32)
    vm_pu = env.net.res_bus['vm_pu'].fillna(0).values.astype(np.float32)

    discrete_switches = np.concatenate([
        env.net.switch['closed'].astype(int).values,
        env.net.line['in_service'].astype(int).values
    ])

    # Build base observation with grid state
    observation = {
        "discrete_switches": discrete_switches,
        "continuous_vm_bus": vm_pu,
        "continuous_sgen_data": env.sgen_data.values[env.time_step].astype(np.float32),
        "continuous_load_data": env.load_data.values[env.time_step].astype(np.float32),
        "continuous_line_loadings": loading_percent,
        "continuous_space_ext_grid_p_mw": env.net.res_ext_grid['p_mw'].fillna(0).values.astype(np.float32),
        "continuous_space_ext_grid_q_mvar": env.net.res_ext_grid['q_mvar'].fillna(0).values.astype(np.float32)
    }

    # Add BESS observations if BESS exists in the environment
    # Backward compatibility: ENV_RHV doesn't have BESS, ENV_BESS does
    if hasattr(env, 'bess_soc') and hasattr(env, 'bess_power'):
        # State of Charge (SoC) - Normalized energy level for each BESS
        # - Agent needs SoC to make informed dispatch decisions
        # - Enables multi-step planning: "Can I discharge now and still have energy for peak later?"
        # - Prevents SoC violations: Agent learns not to discharge when SoC is low
        # - Critical for temporal reasoning: Links current action to future consequences
        observation["bess_soc"] = env.bess_soc.astype(np.float32)

        # Current Power Output - Actual power flow for each BESS
        # - Agent observes effects of its previous action (action feedback)
        # - Enables smooth control: Agent sees if power ramped as intended
        # - Temporal consistency: Links observation(t) to action(t-1)
        # - Helps detect grid constraints: If commanded 50 MW but observe 30 MW
        observation["bess_power"] = env.bess_power.astype(np.float32)

    return observation


# ==================== Step Helpers ====================

def apply_bess_action(env, action):
    """
    Apply continuous power actions to BESS units in the Pandapower network.

    This function modifies the Pandapower network by creating or updating controllable
    generator (sgen) elements that represent BESS power injection/absorption. Each BESS
    action corresponds to a power setpoint that will be used in the subsequent AC power
    flow calculation.

    Sign Convention (matches Pandapower sgen):
    - Positive action (+MW): BESS discharging → power injection to grid (p_mw > 0)
    - Negative action (-MW): BESS charging → power absorption from grid (p_mw < 0)
    - Zero action (0 MW): BESS idle/standby (p_mw = 0)

    Why Use Sgen (Static Generator) for BESS:
    - Sgen elements in Pandapower represent controllable power sources/sinks
    - Can have both positive (generation) and negative (consumption) p_mw
    - Bidirectional power flow matches BESS charge/discharge behavior
    - Simpler than using separate gen (discharge) and load (charge) elements
    - Allows direct control of active power setpoint (perfect for RL actions)

    Network Modification Process:
    1. First call: Creates new sgen elements at BESS bus locations
       - Stores sgen indices in env.bess_sgen_indices for future updates
    2. Subsequent calls: Updates p_mw of existing sgen elements
       - Faster than creating new elements each step
    3. Sets q_mvar = 0 (assumes unity power factor, purely active power control)

    Action Validation:
    - Checks action shape matches (num_bess,)
    - Clips actions to [-bess_power_mw, +bess_power_mw] if exceeded
    - Warns if clipping occurs (indicates agent exceeded physical limits)

    Example Usage:
        # 5 BESS units with 50 MW rating
        action = np.array([-50.0, 0.0, 50.0, 30.0, -40.0])
        apply_bess_action(env, action)

        # Resulting network modifications:
        # - BESS_0 at bus 15: p_mw = -50.0 (charging at max, absorbing 50 MW)
        # - BESS_1 at bus 23: p_mw = 0.0 (idle, no power exchange)
        # - BESS_2 at bus 47: p_mw = 50.0 (discharging at max, injecting 50 MW)
        # - BESS_3 at bus 62: p_mw = 30.0 (partial discharge, injecting 30 MW)
        # - BESS_4 at bus 81: p_mw = -40.0 (charging at 80% rate, absorbing 40 MW)

        # After pp.runpp(env.net), results available in:
        # env.net.res_sgen.loc[env.bess_sgen_indices, 'p_mw']  # Actual power dispatch
        # env.net.res_sgen.loc[env.bess_sgen_indices, 'q_mvar']  # Reactive power (≈0)

    Parameters:
        env: Environment instance with BESS configuration:
            - env.num_bess: Number of BESS units
            - env.bess_power_mw: Maximum charge/discharge power (MW)
            - env.bess_locations: Bus indices where BESS are connected
            - env.net: Pandapower network object
            - env.bess_sgen_indices (optional): Existing sgen indices to update

        action: NumPy array of power setpoints, shape (num_bess,)
            - Units: MW (megawatts)
            - Range: [-bess_power_mw, +bess_power_mw] per unit
            - Positive: discharge, Negative: charge, Zero: idle

    Raises:
        ValueError: If action shape doesn't match (num_bess,)

    Note:
        - This function modifies env.net in-place
        - Power flow (pp.runpp) must be called separately after this function
        - SoC updates are handled separately in update_bess_soc()
    """
    # Validate action shape
    action = np.array(action, dtype=np.float32)
    if action.shape != (env.num_bess,):
        raise ValueError(
            f"Action shape {action.shape} does not match expected shape ({env.num_bess},)"
        )

    # Clip actions to physical power limits
    # - Actions beyond [-bess_power_mw, +bess_power_mw] are physically infeasible
    # - Clipping prevents simulation errors and guides agent to learn valid actions
    # - Warning alerts if agent is trying to exceed limits (useful for debugging)
    clipped_action = np.clip(action, -env.bess_power_mw, env.bess_power_mw)
    if not np.allclose(action, clipped_action):
        exceeded_indices = np.where(~np.isclose(action, clipped_action))[0]
        print(f"Warning: Actions clipped for BESS units {exceeded_indices}")
        print(f"  Original: {action[exceeded_indices]}")
        print(f"  Clipped:  {clipped_action[exceeded_indices]}")
        print(f"  Limit: ±{env.bess_power_mw} MW")

    # Check if BESS sgen elements already exist (from previous step)
    if not hasattr(env, 'bess_sgen_indices') or env.bess_sgen_indices is None:
        # First call: Create new sgen elements for each BESS unit
        # - Controllable generators (sgen) allow bidirectional power flow
        # - p_mw can be positive (discharge) or negative (charge)
        # - This matches BESS behavior perfectly (vs. using separate gen/load)
        env.bess_sgen_indices = []

        for i in range(env.num_bess):
            sgen_idx = pp.create_sgen(
                env.net,
                bus=env.bess_locations[i],      # Bus where BESS is connected
                p_mw=clipped_action[i],         # Power setpoint from action
                q_mvar=0.0,                     # Assume unity power factor (pure P control)
                name=f"BESS_{i}",               # Identifier for tracking
                type="BESS",                    # Type tag for filtering
                in_service=True                 # Active in power flow calculation
            )
            env.bess_sgen_indices.append(sgen_idx)

        env.bess_sgen_indices = np.array(env.bess_sgen_indices, dtype=np.int32)
        print(f"Created {env.num_bess} BESS sgen elements at indices {env.bess_sgen_indices}")

    else:
        # Subsequent calls: Update existing sgen elements
        # - Much faster than creating new elements each step
        # - Only updates p_mw (power setpoint), other parameters unchanged
        for i, sgen_idx in enumerate(env.bess_sgen_indices):
            env.net.sgen.at[sgen_idx, 'p_mw'] = clipped_action[i]

    # Store current actions for observation (will be read in build_observation)
    # - Agent observes executed actions for temporal consistency
    # - Helps learn smooth control policies (avoid sudden power jumps)
    env.bess_power = clipped_action.copy()


def update_bess_soc(env):
    """
    Update BESS State of Charge (SoC) based on actual power flow and energy balance physics.

    This function calculates the energy exchanged by each BESS unit during the timestep
    and updates the SoC accordingly. It uses the actual power output from the AC power
    flow results (not the commanded action) to account for grid constraints and losses.

    Energy Balance Physics:

    The fundamental relationship is: Energy = Power × Time

    For BESS, the change in stored energy depends on:
    1. Actual power exchange with the grid (P, in MW)
    2. Duration of the timestep (Δt, in hours)
    3. Round-trip efficiency (η, dimensionless)
    4. Battery capacity (C, in MWh)

    Energy Balance Equations:

    Discharging (P > 0, energy flows OUT of battery):
    - Energy delivered to grid: E_out = P × Δt (MWh)
    - Energy taken from battery: E_battery = E_out / η
    - SoC decreases: ΔSoC = -E_battery / C = -(P × Δt) / (η × C)
    - Efficiency η < 1 means more energy is taken from battery than delivered to grid

    Charging (P < 0, energy flows INTO battery):
    - Energy taken from grid: E_in = |P| × Δt (MWh)
    - Energy stored in battery: E_battery = E_in × η
    - SoC increases: ΔSoC = +E_battery / C = (|P| × Δt × η) / C
    - Efficiency η < 1 means less energy is stored than taken from grid

    Unified Formula:
    For implementation, we use a unified formula that handles both cases:

        ΔSoC = (P × Δt × efficiency_factor) / capacity

    Where efficiency_factor = {
        η         if P > 0 (discharging, apply losses to energy removed)
        1/η       if P < 0 (charging, apply losses to energy stored)
    }

    Round-Trip Efficiency:
    - If we discharge E_out energy and then charge it back:
      * Discharge: E_battery decreases by E_out / η
      * Charge: E_battery increases by E_out × η
      * Net: E_battery decreases by E_out × (1 - η²)
    - For η = 0.9: round-trip efficiency = 0.9² = 81% (matches real Li-ion)

    Why Use Actual Power (res_sgen) Not Commanded Action:
    - Grid constraints may limit actual power (e.g., voltage violations)
    - Commanded action may be infeasible due to network state
    - res_sgen gives the power that actually flowed through the grid
    - Using actual power ensures energy balance consistency with power flow

    Example Calculation:
        BESS Configuration:
        - Capacity: 50 MWh
        - Efficiency: 90% (η = 0.9)
        - Timestep: 1 hour (Δt = 1)
        - Current SoC: 50% (0.5)

        Case 1: Discharging at 50 MW
        - Actual power from res_sgen: P = +50 MW
        - Energy delivered to grid: 50 MW × 1 h = 50 MWh
        - Energy removed from battery: 50 / 0.9 = 55.56 MWh (losses!)
        - ΔSoC = -55.56 / 50 = -1.111 → clipped to -0.4 (SoC can't go below 10%)
        - New SoC: 0.5 - 0.4 = 0.1 (10%, at lower bound)

        Case 2: Charging at 50 MW
        - Actual power from res_sgen: P = -50 MW
        - Energy taken from grid: 50 MW × 1 h = 50 MWh
        - Energy stored in battery: 50 × 0.9 = 45 MWh (losses!)
        - ΔSoC = +45 / 50 = +0.9
        - New SoC: 0.5 + 0.9 = 1.4 → clipped to 0.9 (SoC can't exceed 90%)

        Case 3: Partial discharge at 20 MW
        - Actual power: P = +20 MW
        - Energy removed: (20 × 1) / 0.9 = 22.22 MWh
        - ΔSoC = -22.22 / 50 = -0.444
        - New SoC: 0.5 - 0.444 = 0.056 → clipped to 0.1 (below soc_min)

    Parameters:
        env: Environment instance with BESS configuration and power flow results:
            - env.bess_soc: Current SoC array (shape: num_bess,)
            - env.bess_sgen_indices: Sgen indices for BESS units
            - env.net.res_sgen: Power flow results with actual p_mw values
            - env.bess_capacity_mwh: Energy capacity per unit (MWh)
            - env.time_step_hours: Timestep duration (hours)
            - env.efficiency: Round-trip efficiency (0 < η ≤ 1)
            - env.soc_min, env.soc_max: SoC constraints (e.g., 0.1, 0.9)

    Updates:
        env.bess_soc: Updated SoC array (clipped to [soc_min, soc_max])
        env.bess_power: Updated with actual power from res_sgen

    Note:
        - This function must be called AFTER pp.runpp() (power flow calculation)
        - SoC clipping enforces physical constraints (agent learns these implicitly)
        - If commanded action violates SoC limits, actual power may be reduced by grid
    """
    # Read actual power output from power flow results
    # - res_sgen contains the power that actually flowed through the grid
    # - May differ from commanded action due to grid constraints (voltage, thermal limits)
    # - Using actual power ensures energy balance consistency
    actual_power = env.net.res_sgen.loc[env.bess_sgen_indices, 'p_mw'].values

    # Calculate SoC change for each BESS unit
    delta_soc = np.zeros(env.num_bess, dtype=np.float32)

    for i in range(env.num_bess):
        P = actual_power[i]  # MW (positive = discharge, negative = charge)

        # Apply efficiency based on charge/discharge direction
        # Round-trip efficiency concept:
        # - Discharging: Energy out of battery > Energy to grid (losses in conversion)
        # - Charging: Energy into battery < Energy from grid (losses in storage)
        if P > 0:
            # DISCHARGING (P > 0): Battery LOSES energy
            # Energy delivered to grid: E_grid = P × Δt = 50 × 1 = 50 MWh
            # Energy removed from battery: E_battery = E_grid / η = 50 / 0.9 = 55.6 MWh
            # Battery loses MORE than it delivers (conversion losses)
            # ΔE_battery = -55.6 MWh (negative = decrease)
            #
            # Formula: ΔE = -(P × Δt / η)
            # Example: P=50, Δt=1, η=0.9 → ΔE = -(50×1/0.9) = -55.6 MWh ✓
            energy_change = -P * env.time_step_hours / env.efficiency

        elif P < 0:
            # CHARGING (P < 0): Battery GAINS energy
            # P = -50 MW means 50 MW flowing INTO battery from grid
            # Energy taken from grid: E_grid = |P| × Δt = 50 × 1 = 50 MWh
            # Energy stored in battery: E_battery = E_grid × η = 50 × 0.9 = 45 MWh
            # Battery stores LESS than taken from grid (storage losses)
            # ΔE_battery = +45 MWh (positive = increase)
            #
            # Formula: ΔE = |P| × Δt × η = -P × Δt × η (since P is negative, -P is positive)
            # Example: P=-50, Δt=1, η=0.9 → ΔE = -(-50)×1×0.9 = 50×0.9 = +45 MWh ✓
            energy_change = -P * env.time_step_hours * env.efficiency  # -P converts negative to positive

        else:
            # Idle: No power exchange, no SoC change
            energy_change = 0.0

        # Convert energy change (MWh) to SoC change (fraction)
        # SoC is normalized: 0.0 = empty (0 MWh), 1.0 = full (capacity MWh)
        # ΔSoC = ΔEnergy / Capacity
        # Units: MWh / MWh = dimensionless (fraction)
        #
        # Examples:
        # - Discharge: ΔE = -55.6 MWh, C = 50 MWh → ΔSoC = -55.6/50 = -1.11 (decrease)
        # - Charge: ΔE = +45 MWh, C = 50 MWh → ΔSoC = +45/50 = +0.9 (increase)
        delta_soc[i] = energy_change / env.bess_capacity_mwh

    # Update SoC with calculated changes
    new_soc = env.bess_soc + delta_soc

    # Enforce SoC constraints
    # - soc_min (e.g., 0.1 = 10%): Prevents deep discharge, extends battery life
    # - soc_max (e.g., 0.9 = 90%): Prevents overcharge, maintains safety margins
    # - Clipping teaches agent the operational bounds (via reward penalties when hitting limits)
    clipped_soc = np.clip(new_soc, env.soc_min, env.soc_max)

    # Check if any SoC values were clipped (hit constraints)
    if not np.allclose(new_soc, clipped_soc):
        clipped_indices = np.where(~np.isclose(new_soc, clipped_soc))[0]
        for idx in clipped_indices:
            if new_soc[idx] < env.soc_min:
                print(f"Warning: BESS {idx} SoC hit lower bound ({env.soc_min*100:.0f}%)")
                print(f"  Attempted: {new_soc[idx]*100:.1f}%, Clipped to: {env.soc_min*100:.0f}%")
            elif new_soc[idx] > env.soc_max:
                print(f"Warning: BESS {idx} SoC hit upper bound ({env.soc_max*100:.0f}%)")
                print(f"  Attempted: {new_soc[idx]*100:.1f}%, Clipped to: {env.soc_max*100:.0f}%")

    # Store updated SoC
    env.bess_soc = clipped_soc

    # Update env.bess_power with actual power from power flow
    # - This reflects what actually happened (vs. what was commanded)
    # - Will be included in the next observation for agent feedback
    env.bess_power = actual_power.astype(np.float32)

    # Optional: Print debug info
    # print(f"SoC update: P={actual_power} MW, ΔSoC={delta_soc*100:.2f}%, New SoC={clipped_soc*100:.1f}%")


def calculate_bess_reward(env, max_loading_before, max_loading_after):
    """
    Calculate reward for BESS dispatch actions focused on congestion management.

    This function computes a multi-objective reward signal that primarily incentivizes
    reducing grid congestion (line overloading) while maintaining operational sustainability
    through secondary objectives (SoC management, energy efficiency).

    Reward Structure (3 components):

    1. PRIMARY: Congestion Relief (Dominant, ~10× larger than others)
       - Objective: Reduce maximum line loading percentage
       - Formula: R_congestion = bonus_constant × (loading_before - loading_after)
       - Rationale: This is the thesis's main goal - use BESS to alleviate congestion
       - Why max loading: Worst-case congestion is the critical grid constraint
       - Weight: ~10.0 (default bonus_constant)

    2. SECONDARY: SoC Boundary Penalties (Medium, ~10× smaller than congestion)
       - Objective: Prevent BESS from getting stuck near SoC limits
       - Formula: R_soc = -soc_penalty_weight × num_units_near_bounds
       - Near bounds: SoC < (soc_min + 5%) OR SoC > (soc_max - 5%)
       - Rationale: Maintains operational flexibility for future dispatch
       - Why needed: Without this, agent might discharge to 10% and get stuck
       - Weight: ~-1.0

    3. TERTIARY: Energy Efficiency (Small, ~100× smaller than congestion)
       - Objective: Encourage smooth operation, battery longevity
       - Formula: R_efficiency = -efficiency_weight × Σ(|power_i| / max_power)²
       - Rationale: Extreme charge/discharge rates degrade battery faster
       - Why quadratic: Penalizes full-power operation more than partial
       - Weight: ~-0.1

    Weight Hierarchy Rationale:
    - Congestion (10.0) >> SoC (-1.0) >> Efficiency (-0.1)
    - This ensures agent prioritizes thesis goal (congestion) while learning sustainable operation
    - If weights were equal, agent might preserve battery instead of helping grid
    - If SoC penalty too high, agent becomes overly conservative

    Example Calculations:

    Scenario 1: Good Action - Reduces Congestion, Maintains Flexibility
    - Before: max_loading = 120%
    - After: max_loading = 95% (reduced by 25%)
    - BESS state: 3 units, SoCs = [0.45, 0.55, 0.60], powers = [20, -10, 30] MW (max=50)
    - Congestion reward: 10.0 × (120 - 95) = +250.0 ✓ Large positive
    - SoC penalty: 0 units near bounds (all in 0.15-0.85 safe zone) = 0.0 ✓ No penalty
    - Efficiency: -0.1 × ((20/50)² + (10/50)² + (30/50)²) = -0.1 × 0.53 = -0.053 ✓ Small
    - TOTAL: 250.0 + 0.0 - 0.053 = +249.95 → EXCELLENT ACTION

    Scenario 2: Bad Action - Helps Congestion but Depletes Battery
    - Before: max_loading = 120%
    - After: max_loading = 100% (reduced by 20%)
    - BESS state: 3 units, SoCs = [0.12, 0.11, 0.88], powers = [50, 50, -50] MW
    - Congestion reward: 10.0 × (120 - 100) = +200.0 ✓ Good
    - SoC penalty: 3 units near bounds (0.12 < 0.15, 0.11 < 0.15, 0.88 > 0.85) = -1.0 × 3 = -3.0 ✗ Penalty
    - Efficiency: -0.1 × (1² + 1² + 1²) = -0.1 × 3 = -0.3 ✗ Max power usage
    - TOTAL: 200.0 - 3.0 - 0.3 = +196.7 → SUBOPTIMAL (lower than Scenario 1)

    Scenario 3: Neutral Action - No Congestion Change
    - Before: max_loading = 95%
    - After: max_loading = 95% (no change)
    - BESS state: 3 units, SoCs = [0.50, 0.50, 0.50], powers = [0, 0, 0] MW (idle)
    - Congestion reward: 10.0 × (95 - 95) = 0.0 ⊘ Neutral
    - SoC penalty: 0 units near bounds = 0.0 ✓
    - Efficiency: 0.0 (no power) = 0.0 ✓
    - TOTAL: 0.0 → NEUTRAL (no help, no harm)

    Scenario 4: Very Bad Action - Worsens Congestion
    - Before: max_loading = 95%
    - After: max_loading = 110% (increased by 15% - wrong direction!)
    - BESS state: irrelevant (action was harmful)
    - Congestion reward: 10.0 × (95 - 110) = -150.0 ✗ Large negative
    - SoC penalty: assume -2.0 ✗
    - Efficiency: assume -0.2 ✗
    - TOTAL: -150.0 - 2.0 - 0.2 = -152.2 → VERY BAD ACTION

    Parameter Tuning Guidance:
    - bonus_constant (default 10.0):
      * Increase if agent ignores congestion → makes congestion relief more valuable
      * Decrease if agent is too aggressive → reduces congestion reward magnitude
    - soc_penalty_weight (default -1.0):
      * Increase magnitude if agent gets stuck at bounds → stronger discouragement
      * Decrease magnitude if agent is too conservative → allows more aggressive dispatch
    - efficiency_penalty_weight (default -0.1):
      * Increase magnitude to encourage smoother operation
      * Usually keep small (battery longevity is secondary to grid operation)

    Relationship to Thesis Goal:
    - Thesis: "BESS for congestion management in HV distribution grids"
    - This reward directly optimizes that: max weight on (loading_before - loading_after)
    - SoC and efficiency penalties ensure sustainable operation (not one-shot solutions)
    - Agent learns: "Reduce congestion while maintaining long-term operational capability"

    Parameters:
        env: Environment instance with:
            - env.bess_soc: Current SoC array (shape: num_bess,)
            - env.bess_power: Current power array (shape: num_bess,)
            - env.bess_power_mw: Maximum power rating (MW)
            - env.soc_min, env.soc_max: SoC constraints
            - env.bonus_constant (optional): Congestion reward weight (default 10.0)
            - env.soc_penalty_weight (optional): SoC penalty weight (default -1.0)
            - env.efficiency_penalty_weight (optional): Efficiency penalty (default -0.1)

        max_loading_before: Maximum line loading before action (%)
        max_loading_after: Maximum line loading after action (%)

    Returns:
        tuple: (total_reward, reward_breakdown)
            - total_reward (float): Sum of all reward components
            - reward_breakdown (dict): Individual components for logging/debugging
              {'congestion': float, 'soc_penalty': float, 'efficiency_penalty': float}
    """
    # ========== 1. PRIMARY: Congestion Relief Reward ==========
    # This is the main thesis objective: reduce grid congestion using BESS
    # - Positive loading reduction → positive reward (good action)
    # - Negative loading reduction (worsening) → negative reward (bad action)
    # - Zero change → zero reward (neutral action)
    #
    # Why use max_loading not average:
    # - Grid operators care about worst-case (preventing equipment damage)
    # - Single overloaded line can cause cascading failures
    # - Max loading directly relates to grid security
    bonus_constant = getattr(env, 'bonus_constant', 10.0)
    congestion_reward = bonus_constant * (max_loading_before - max_loading_after)

    # ========== 2. SECONDARY: SoC Boundary Penalties ==========
    # Penalize operating too close to SoC limits (prevents getting stuck)
    # - If SoC < (soc_min + 5%): Can't discharge much → limited future flexibility
    # - If SoC > (soc_max - 5%): Can't charge much → can't absorb excess generation
    # - Agent learns to maintain "operational headroom" for future dispatch
    soc_penalty_weight = getattr(env, 'soc_penalty_weight', -1.0)
    boundary_margin = getattr(env, 'soc_boundary_margin', 0.05)  # 5% margin from bounds

    num_near_lower = np.sum(env.bess_soc < (env.soc_min + boundary_margin))
    num_near_upper = np.sum(env.bess_soc > (env.soc_max - boundary_margin))
    num_near_bounds = num_near_lower + num_near_upper

    soc_penalty = soc_penalty_weight * num_near_bounds

    # ========== 3. TERTIARY: Energy Efficiency Penalty ==========
    # Encourage partial power dispatch over full-power operation
    # - Extreme charge/discharge rates degrade battery faster (real-world concern)
    # - Quadratic penalty: 100% power penalized 4× more than 50% power
    # - Agent learns smooth, sustainable operation when possible
    #
    # Weight balance: This is 100× smaller than congestion reward
    # - Congestion management is primary, efficiency is secondary
    # - Only influences decisions when congestion impact is similar
    
    
    # DISABLED: Efficiency penalty was preventing BESS usage
    # efficiency_penalty_weight = getattr(env, 'efficiency_penalty_weight', -0.1)
    # normalized_power = np.abs(env.bess_power) / env.bess_power_mw
    # efficiency_penalty = efficiency_penalty_weight * np.sum(normalized_power ** 2)
    
    # NEW: Action magnitude bonus (encourages BESS usage)
    action_magnitude = np.mean(np.abs(env.bess_power) / env.bess_power_mw)
    action_bonus = 5.0 * action_magnitude

    # ========== Total Reward ==========
    #total_reward = congestion_reward + soc_penalty + efficiency_penalty (replaced below)

    total_reward = congestion_reward + soc_penalty + action_bonus

    # Breakdown for logging/debugging
    reward_breakdown = {
        'congestion': float(congestion_reward),
        'soc_penalty': float(soc_penalty),
        'efficiency_penalty': float(action_bonus)  # Changed name to action_bonus
    }

    return total_reward, reward_breakdown


def validate_grid_state_after_action(env):
    """Validate grid state after action and return error result if invalid."""
    # Run load flow calculations
    if not run_power_flow(env, "stepping"):
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


# ==================== Update State Helpers ====================

def update_to_next_timestep(env):
    """Update environment to the next timestep."""
    env.time_step += 1
    if env.time_step >= env.sgen_data.shape[0]:
        env.time_step = 0

    env.net = copy.deepcopy(env.initial_net)
    env.relative_index = env.sgen_data.index.values[env.time_step]
    env.net = apply_absolute_values_to_network(env.net, env.profiles, env.relative_index)

    # Re-create BESS sgen elements AFTER profile update
    if hasattr(env, 'bess_sgen_indices') and env.bess_sgen_indices is not None:
        env.bess_sgen_indices = None
        apply_bess_action(env, env.bess_power)

    # Run load flow calculations
    if not run_power_flow(env, "updating"):
        pass  # Error already logged in run_power_flow
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