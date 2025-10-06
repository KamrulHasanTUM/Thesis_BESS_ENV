<div align="center">

# ENV_BESS

### Battery Energy Storage System Reinforcement Learning Environment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PowerGrid](https://img.shields.io/badge/Application-Power%20Grid-red.svg)](https://github.com/KamrulHasanTUM/Thesis_BESS_ENV)

**A state-of-the-art Gymnasium-compatible RL environment for intelligent congestion management in high-voltage distribution grids using Battery Energy Storage Systems**

[Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#environment-details) • [Research](#research-context) • [Citation](#citation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Research Context](#research-context)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Author](#author)
- [Support](#support)

---

## Overview

ENV_BESS is a cutting-edge reinforcement learning environment designed to tackle one of the most critical challenges in modern power systems: grid congestion management. By integrating battery energy storage systems (BESS) with advanced RL algorithms, this environment enables intelligent, real-time decision-making for optimal power dispatch.

### What Makes ENV_BESS Special?

- **Real-World Physics**: SimBench benchmark networks (110 kV), Pandapower AC power flow simulation, Physics-based BESS energy balance, 35,136 hourly timesteps (~4 years data)
- **RL-Ready Design**: Gymnasium API compatibility, Continuous action space (±50 MW), Multi-modal observation space, Validated with 9 comprehensive tests

---

## Key Features

| Feature | Description | Value |
|---------|-------------|-------|
| **Action Space** | Continuous power dispatch | Box(-50, 50) MW |
| **Observations** | Multi-modal grid + BESS state | 9 components |
| **Primary Goal** | Line congestion reduction | Reward-based |
| **BESS Units** | Configurable battery systems | Default: 5 units |
| **Capacity** | Energy storage per unit | 50 MWh |
| **Efficiency** | Round-trip efficiency | 90% |
| **Constraints** | Realistic operational limits | SoC: 10-90% |
| **Validation** | Comprehensive test suite | 9 tests passed |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda/Miniconda (recommended)
- 16GB RAM (for training)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/KamrulHasanTUM/Thesis_BESS_ENV.git
cd Thesis_BESS_ENV

# Create conda environment
conda create -n bess_env python=3.8
conda activate bess_env

# Install dependencies
pip install gymnasium stable-baselines3 pandapower simbench scikit-learn tqdm numpy pandas
```

### Verify Installation

```bash
cd tests
python run_all_tests.py
```

Expected Output:
```
Total tests: 9
Passed: 9
Failed: 0
[PASS] ALL TESTS PASSED - Environment is ready for training!
```

---

## Quick Start

### Configure Your Experiment

Create `init_meta.json`:

```json
{
  "exp_code": "bess_test",
  "exp_id": 1,
  "exp_name": "BESS Congestion Management",
  "grid_env": "bess"
}
```

### Train an Agent

```python
from ENV_BESS_main import ENV_BESS
from stable_baselines3 import PPO

# Create environment
env = ENV_BESS(num_bess=5, bess_power_mw=50.0, max_step=50)

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("bess_ppo_model")
```

### Evaluate Performance

```python
# Load trained model
model = PPO.load("bess_ppo_model")

# Run evaluation episode
obs, info = env.reset()
total_reward = 0

for step in range(50):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward:.2f}")
```

---

## Environment Details

### Action Space

**Type:** `Box(low=-50, high=50, shape=(5,), dtype=float32)`

Each BESS unit receives a continuous power setpoint:

- **Negative values** → Charging (absorb power from grid)
- **Positive values** → Discharging (inject power to grid)
- **Range:** ±50 MW per unit

### Observation Space

**Type:** Dict with 9 components

| Component | Description | Shape | Range |
|-----------|-------------|-------|-------|
| `bess_soc` | State of Charge (normalized) | `(5,)` | [0.0, 1.0] |
| `bess_power` | Current power output | `(5,)` | [-50, 50] MW |
| `continuous_vm_bus` | Bus voltages | `(buses,)` | [0.5, 1.5] p.u. |
| `continuous_line_loadings` | Line loading % | `(lines,)` | [0, 800] % |
| `continuous_load_data` | Load consumption | `(loads,)` | [0, 100K] MW |
| `continuous_sgen_data` | Generator output | `(gens,)` | [0, 100K] MW |
| `continuous_space_ext_grid_p_mw` | External grid P | `(1,)` | ±50M MW |
| `continuous_space_ext_grid_q_mvar` | External grid Q | `(1,)` | ±50M MVAr |
| `discrete_switches` | Switch states | `(switches,)` | [0, 1] |

### Reward Function

Multi-objective reward designed to balance congestion relief with BESS sustainability:

```python
R_total = R_congestion + R_soc_penalty + R_efficiency
```

**Component Breakdown:**

| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| **Congestion** | `10.0 × (loading_before - loading_after)` | **10.0** | Primary goal: reduce overloading |
| **SoC Penalty** | `-1.0 × num_units_near_bounds` | **-1.0** | Avoid extreme SoC levels |
| **Efficiency** | `-0.1 × Σ(power/max_power)²` | **-0.1** | Encourage efficient operation |

Design Philosophy: Congestion relief is 10× more important than SoC management, which is 10× more important than efficiency optimization.

### Episode Configuration

- **Length:** 50 timesteps (configurable)
- **Timestep duration:** 1 hour
- **Real-world span:** ~2 days per episode
- **Network:** SimBench 1-HV-mixed (110 kV)

**Termination Conditions:**
- Max steps reached (normal completion)
- Power flow convergence failure
- Excessive line disconnections
- Voltage violations (NaN values)

---

## Configuration

### BESS Parameters

```python
env = ENV_BESS(
    # BESS Configuration
    num_bess=5,                  # Number of battery units
    bess_capacity_mwh=50.0,      # Energy capacity (MWh)
    bess_power_mw=50.0,          # Power rating (MW)
    soc_min=0.1,                 # Minimum SoC (10%)
    soc_max=0.9,                 # Maximum SoC (90%)
    initial_soc=0.5,             # Starting SoC (50%)
    efficiency=0.9,              # Round-trip efficiency (90%)
    time_step_hours=1.0,         # Timestep duration

    # Grid Configuration
    simbench_code="1-HV-mixed--0-sw",
    max_step=50,
    bonus_constant=10.0,         # Congestion reward weight
)
```

### Training Hyperparameters

```python
model = PPO(
    "MultiInputPolicy",
    env,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1
)
```

---

## Project Structure

```
Thesis_BESS_ENV/
├── ENV_BESS_main.py              # Main environment class
├── env_helpers.py                # Helper functions (reset, step, reward)
├── config.py                     # Configuration management
├── training.py                   # Training utilities
├── utils.py                      # Miscellaneous utilities
├── tests/                        # Comprehensive test suite
│   ├── run_all_tests.py          # Master test runner
│   ├── test_config.py            # Configuration tests
│   ├── test_env_creation.py      # Environment creation
│   ├── test_reset.py             # Reset functionality
│   ├── test_episode.py           # Episode execution
│   ├── test_soc_dynamics.py      # Battery physics
│   ├── test_rewards.py           # Reward calculation
│   ├── test_full_episode.py      # Full episode flow
│   ├── test_multiple_episodes.py # Multi-episode stability
│   └── test_gym_api.py           # Gymnasium API compliance
├── init_meta.json.example        # Example configuration
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## Research Context

### Thesis Background

This environment was developed as part of a Master's thesis:

**"Reinforcement Learning for Battery Energy Storage System Based Congestion Management in High-Voltage Distribution Grids"**

### Problem Statement

Modern power grids face unprecedented challenges:

- **Renewable Integration**: Growing solar/wind penetration creates volatile generation patterns
- **EV Adoption**: Rising electric vehicle charging increases peak demand
- **Load Fluctuation**: Unpredictable consumption patterns stress grid infrastructure

**Traditional Solution:** Grid expansion → Costly, Time-consuming, Environmentally impactful

**Our Solution:** BESS-based RL → Fast, Targeted, Flexible

### Approach

| Aspect | Implementation |
|--------|----------------|
| **RL Algorithm** | Proximal Policy Optimization (PPO) |
| **Control Type** | Continuous action space |
| **Objective** | Minimize line overloading + BESS sustainability |
| **Grid Model** | SimBench 1-HV-mixed (110 kV) |
| **Dataset** | 35,136 hourly timesteps (~4 years) |
| **Features** | Multi-modal observations (grid + BESS state) |

### Performance Metrics

**Training Configuration:**
- Hardware: i7 CPU, 16GB RAM (consumer-grade workstation)
- Training time: ~34 hours for 1M timesteps
- Episodes: ~24,400 episodes
- Convergence: Effective policies within 500k timesteps

### Current Limitations

- No battery degradation modeling
- Active power only (no reactive power control)
- Single voltage level (110 kV)
- Assumes perfect power flow convergence

### Future Roadmap

- Multi-voltage level support (HV/MV/LV)
- Battery aging and degradation models
- Reactive power (Q) optimization
- Multi-agent distributed BESS scenarios
- Real-time grid data integration
- Transfer learning across different grids

---

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{hasan2025bess_env,
  author = {Hasan, Kamrul},
  title = {ENV_BESS: RL Environment for BESS-based Grid Congestion Management},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/KamrulHasanTUM/Thesis_BESS_ENV},
  note = {Master's Thesis Project}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project builds upon excellent open-source tools:

| Project | Purpose | Link |
|---------|---------|------|
| **SimBench** | Realistic benchmark power grids | [simbench.de](https://simbench.de) |
| **Pandapower** | AC power flow simulation | [pandapower.org](https://www.pandapower.org) |
| **Stable-Baselines3** | RL algorithm implementations | [GitHub](https://github.com/DLR-RM/stable-baselines3) |
| **Gymnasium** | RL environment standard | [Gymnasium](https://gymnasium.farama.org) |

---

## Author

**Kamrul Hasan**

[![GitHub](https://img.shields.io/badge/GitHub-KamrulHasanTUM-181717?logo=github)](https://github.com/KamrulHasanTUM)

---

## Support

Having issues? Here's how to get help:

1. Check Tests → Run `tests/run_all_tests.py` for diagnostics
2. Review Examples → See usage patterns in code
3. Report Issues → Open a GitHub issue with error messages, environment configuration, and steps to reproduce

---

<div align="center">

**Built with ❤️ for advancing power grid intelligence**

[Back to Top](#env_bess)

</div>
