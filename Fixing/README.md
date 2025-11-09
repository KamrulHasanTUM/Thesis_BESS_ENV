# Diagnostic Test Suite for BESS RL Environment

This folder contains diagnostic tests and improved implementations to fix the RL training issues.

## Test Structure

1. **test_1_random_baseline.py** - Tests if BESS can physically reduce congestion
2. **test_2_zero_action_baseline.py** - Tests grid behavior without BESS
3. **test_3_soc_clipping.py** - Implements and tests SoC-aware action clipping
4. **test_4_simplified_reward.py** - Tests simplified reward function
5. **improved_env_helpers.py** - Fixed reward function and action clipping
6. **test_config.py** - Configuration for diagnostic tests
7. **run_all_tests.py** - Runs all diagnostic tests in sequence

## How to Use

1. Run individual tests:
   ```bash
   python Fixing/test_1_random_baseline.py
   ```

2. Run all tests:
   ```bash
   python Fixing/run_all_tests.py
   ```

3. If tests pass, manually apply changes to original code based on `CHANGES_TO_APPLY.md`

## Test Results

Results will be saved in `Fixing/results/` directory with timestamps.

## DO NOT modify original code until all tests pass!
