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

