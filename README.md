@'

\# ENV\_BESS: Battery Energy Storage System RL Environment



A Gymnasium-compatible reinforcement learning environment for Battery Energy Storage System (BESS) based congestion management in high-voltage distribution grids.



\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\## Overview



ENV\_BESS is a custom RL environment that simulates real-world power grid operations with battery storage systems. It integrates:

\- \*\*SimBench\*\* benchmark networks for realistic grid topology

\- \*\*Pandapower\*\* for AC power flow simulations

\- \*\*Gymnasium\*\* API for RL algorithm compatibility

\- \*\*Physics-based BESS modeling\*\* with energy balance and efficiency



\### Key Features



\- \*\*Continuous action space\*\*: Fine-grained BESS power dispatch control (±50 MW per unit)

\- \*\*Multi-modal observations\*\*: Grid state (voltages, line loadings) + BESS state (SoC, power)

\- \*\*Congestion-aware rewards\*\*: Primary objective is reducing line overloading

\- \*\*Realistic constraints\*\*: SoC limits (10-90%), power ratings, round-trip efficiency

\- \*\*Validated implementation\*\*: 9 comprehensive tests ensuring correctness



\## Installation



\### Prerequisites



\- Python 3.8 or higher

\- Anaconda/Miniconda (recommended)



\### Setup

```bash

\# Clone the repository

git clone https://github.com/KamrulHasanTUM/Thesis\_BESS\_ENV.git

cd Thesis\_BESS\_ENV



\# Create conda environment

conda create -n bess\_env python=3.8

conda activate bess\_env



\# Install dependencies

pip install gymnasium

pip install stable-baselines3

pip install pandapower

pip install simbench

pip install scikit-learn

pip install tqdm

pip install numpy pandas


Quick Start
1. Configure Experiment
Create init_meta.json:

{
    "exp_code": "bess_test",
    "exp_id": 1,
    "exp_name": "BESS Training",
    "grid_env": "bess"
}

2. Run Verification Tests

cd tests
python run_all_tests.py

Expected output:

Total tests: 9
Passed: 9
Failed: 0
[PASS] ALL TESTS PASSED - Environment is ready for training!

3. Train PPO Agent

from ENV_BESS_main import ENV_BESS
from stable_baselines3 import PPO

# Create environment
env = ENV_BESS(
    num_bess=5,
    bess_power_mw=50.0,
    bess_capacity_mwh=50.0,
    max_step=50
)

# Initialize PPO
model = PPO("MultiInputPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=100_000)
model.save("bess_ppo_model")

4. Evaluate Trained Agent

# Load trained model
model = PPO.load("bess_ppo_model")

# Run evaluation episode
obs, info = env.reset()
for _ in range(50):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break


Environment Details
Action Space
Type: Box(low=-50, high=50, shape=(5,), dtype=float32)
Continuous power setpoints for each BESS unit:

Negative values: Charging (absorbing power from grid)
Positive values: Discharging (injecting power to grid)
Units: Megawatts (MW)

Observation Space
# Type: **Dict with 9 components**

| Key                                | Description                     | Shape           | Range            |
|------------------------------------|---------------------------------|-----------------|------------------|
| **bess_soc**                       | State of Charge (normalized)    | (5,)            | [0.0, 1.0]       |
| **bess_power**                     | Current power output           | (5,)            | [-50, 50] MW     |
| **continuous_vm_bus**              | Bus voltages                   | (num_buses,)    | [0.5, 1.5] p.u.  |
| **continuous_line_loadings**       | Line loading %                 | (num_lines,)    | [0, 800] %       |
| **continuous_load_data**           | Load power                     | (num_loads,)    | [0, 100000] MW   |
| **continuous_sgen_data**           | Generator power                | (num_sgen,)     | [0, 100000] MW   |
| **continuous_space_ext_grid_p_mw** | External grid P                | (1,)            | ±50M MW          |
| **continuous_space_ext_grid_q_mvar** | External grid Q              | (1,)            | ±50M MVAr        |
| **discrete_switches**              | Switches                       | (num_switches,) | [0, 1]           |


Reward Function
Multi-objective reward balancing congestion relief with operational sustainability:
R_total = R_congestion + R_soc_penalty + R_efficiency

where:
  R_congestion = bonus_constant × (loading_before - loading_after)
  R_soc_penalty = -1.0 × num_units_near_bounds
  R_efficiency = -0.1 × Σ(|power_i| / max_power)²
Weights: Congestion (10.0) >> SoC (-1.0) >> Efficiency (-0.1)

Episode Configuration

Length: 50 timesteps (configurable)
Timestep duration: 1 hour
Real-world span: ~2 days per episode
Termination conditions:

Max steps reached
Power flow convergence failure
Excessive line disconnections
Voltage violations

Configuration
Edit config.py to adjust hyperparameters:
BESS Parameters
'num_bess': 5,              # Number of BESS units
'bess_capacity_mwh': 50.0,  # Energy capacity per unit
'bess_power_mw': 50.0,      # Power rating per unit
'soc_min': 0.1,             # Minimum SoC (10%)
'soc_max': 0.9,             # Maximum SoC (90%)
'initial_soc': 0.5,         # Starting SoC (50%)
'efficiency': 0.9,          # Round-trip efficiency

Training Parameters
'n_epochs': 10,
'n_steps': 2048,
'batch_size': 256,
'total_timesteps': 1_000_000,
'learning_rate': 0.0003,

Project Structure
Thesis_BESS_ENV/
├── ENV_BESS_main.py          # Main environment class
├── env_helpers.py            # Helper functions (reset, step, reward, etc.)
├── config.py                 # Configuration management
├── training.py               # Training utilities
├── utils.py                  # Miscellaneous utilities
├── tests/                    # Verification tests
│   ├── run_all_tests.py      # Master test runner
│   ├── test_config.py
│   ├── test_env_creation.py
│   ├── test_reset.py
│   ├── test_episode.py
│   ├── test_soc_dynamics.py
│   ├── test_rewards.py
│   ├── test_full_episode.py
│   ├── test_multiple_episodes.py
│   └── test_gym_api.py
├── init_meta.json.example    # Example configuration
├── README.md
└── LICENSE

Development
Running Tests
# All tests
cd tests && python run_all_tests.py

# Individual test
python tests/test_soc_dynamics.py

Code Quality
The codebase follows:

PEP 8 style guidelines
Comprehensive docstrings
Type hints where applicable
Modular architecture for maintainability

Research Context
This environment was developed as part of a Master's thesis on:
"Reinforcement Learning for Battery Energy Storage System Based Congestion Management in High-Voltage Distribution Grids"
Problem Statement
Modern distribution grids face increasing congestion due to:

Growing renewable energy integration
Rising electric vehicle adoption
Fluctuating demand patterns

Traditional grid expansion is costly and time-consuming. BESS offers a flexible alternative for congestion management through optimal charge/discharge scheduling.
Approach

RL Framework: PPO algorithm for continuous control
Objective: Minimize line overloading while maintaining BESS operational health
Network: SimBench 1-HV-mixed benchmark (110 kV distribution grid)
Data: 35,136 hourly timesteps (~4 years) of realistic load/generation profiles

Performance
Training on a standard workstation (i7 CPU, 16GB RAM):

Training time: ~34 hours for 1M timesteps
Episodes: ~24,400 episodes
Convergence: Agent typically learns effective policies within 500k timesteps

Limitations

Assumes perfect power flow convergence
Does not model battery degradation over time
Reactive power control not implemented
Single voltage level (110 kV)

Future Work

 Multi-voltage level support
 Battery aging models
 Reactive power optimization
 Multi-agent scenarios (distributed BESS)
 Real-time grid data integration

Citation
If you use this environment in your research, please cite:
@software{hasan2025bess_env,
  author = {Hasan, Kamrul},
  title = {ENV_BESS: RL Environment for BESS-based Grid Congestion Management},
  year = {2025},
  url = {https://github.com/KamrulHasanTUM/Thesis_BESS_ENV}
}

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

SimBench: Benchmark dataset for power system simulation
Pandapower: Power flow calculation engine
Stable-Baselines3: RL algorithm implementations
Gymnasium: RL environment standard

Contact
Kamrul Hasan

GitHub: @KamrulHasanTUM
Repository: Thesis_BESS_ENV

Support
If you encounter issues:

Check the tests/ directory for validation
Review examples/ for usage patterns
Open an issue with detailed error messages
'@ | Out-File -FilePath README.md -Encoding UTF8

## Step 3: Create LICENSE (MIT)
```powershell
@'
MIT License

Copyright (c) 2025 Kamrul Hasan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'@ | Out-File -FilePath LICENSE -Encoding UTF8