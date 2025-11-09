# Diagnostic Tests Folder

This folder contains all diagnostic test files and reports. **These files will NOT impact your simulation or training.**

## Folder Contents

### Test Scripts

1. **policy_diagnostic_test.py** - **Main diagnostic test (USE THIS ONE)**
   - Works with trained policy OR random policy
   - Gives 100% certainty about:
     - Congestion reduction (loading before vs after)
     - Agent learning quality (responsive vs blind actions)
     - BESS operation (all 5 units working)
     - Reward components accuracy
     - SoC boundary violations

2. **intensive_diagnostic_test.py** - Previous baseline test (random policy only)

### Reports

- **CONFIG_V3_DIAGNOSTIC_REPORT.txt** - Comprehensive analysis of Config V3
- **QUICK_SUMMARY_CONFIG_V3.txt** - Quick reference summary
- **BEFORE_AFTER_COMPARISON_REPORT.txt** - Config V1 vs V2 vs V3 comparison
- **DIAGNOSTIC_EXECUTIVE_SUMMARY.txt** - Executive summary from first test

### Results

- **results/** - Folder where test results are saved (JSON format)
- **diagnostic_results_*.json** - Previous test results

## How to Use the Diagnostic Test

### Option 1: Test with Trained Policy (RECOMMENDED)

After you train your agent (with `training.py` or similar), run:

```bash
cd "D:\Thesis\ThesisEnv_refactor\diagnostic_tests"
python policy_diagnostic_test.py --model ../final_model.zip
```

This will:
- Load your trained policy
- Test it for 10 episodes (500 steps)
- Show you exactly how well it's learning
- Give detailed congestion reduction analysis
- Verify all 5 BESS units are working
- Check reward balance and SoC management

### Option 2: Baseline Test with Random Policy

To establish a baseline BEFORE training:

```bash
cd "D:\Thesis\ThesisEnv_refactor\diagnostic_tests"
python policy_diagnostic_test.py --random
```

### Option 3: Extended Testing

For longer, more thorough tests:

```bash
# Test for 20 episodes instead of 10
python policy_diagnostic_test.py --model ../final_model.zip --episodes 20

# Test with 75 steps per episode instead of 50
python policy_diagnostic_test.py --model ../final_model.zip --steps 75
```

## What the Test Will Tell You

### 1. Congestion Reduction (Your Main Question)

The test tracks loading BEFORE and AFTER every action:

```
Loading Statistics:
  Avg loading BEFORE action: 13.52%
  Avg loading AFTER action: 12.84%
  Net change: -0.68% (REDUCED!)
```

It also shows:
- Improvement steps vs worsening steps
- Improvement rate percentage
- Episode-by-episode performance

### 2. Agent Learning Quality (Your Main Question)

The test analyzes if actions are "blind" or "responsive":

```
Learning Quality:
  Responsiveness Ratio: 0.72

  INTERPRETATION FOR TRAINED POLICY:
    Status: GOOD LEARNING - Agent responding well to grid
    Performance: Significantly better than random
```

For random policy, expect ratio ~0.50 (50/50 chance).
For trained policy, want ratio >0.65 (agent helping more than hurting).

### 3. BESS Operation (Your Question)

Detailed tracking for all 5 units:

```
BESS OPERATION ANALYSIS (ALL 5 UNITS):
  BESS Unit 0:
    Charge: 245 (49.0%)
    Discharge: 223 (44.6%)
    Idle: 32 (6.4%)

  BESS Unit 1:
    Charge: 251 (50.2%)
    ... etc for all 5 units

  Overall BESS Status:
    ALL 5 BESS UNITS ARE OPERATIONAL ✓
```

If any unit shows "NO ACTIVITY DETECTED", you'll know immediately.

### 4. Reward Accuracy (Your Question)

Shows all reward components:

```
Reward Component Analysis:
  Total reward: -28.60
  Congestion reward: -5.20
  SoC penalty: -35.00
  Flexibility bonus: +10.15
  Clipping penalty: -5.82

  Reward Balance Ratio:
    Congestion/SoC ratio: 3.5:1
    Status: WELL BALANCED for learning ✓
```

### 5. SoC Boundary Violations (Your Question)

Detailed violation tracking:

```
Boundary Violations:
  Lower bound (10%) hits: 123
  Upper bound (90%) hits: 45
  Total violations: 168
  Violations per step: 0.34
  Lower/Upper ratio: 2.7:1

  Status: DISCHARGE BIAS - More lower bound hits
```

Also shows episode-by-episode violation trends.

## Understanding the Results

### For Random Policy (Baseline)

Expected results:
- Improvement rate: ~0% (random can't systematically improve)
- Responsiveness ratio: ~0.50 (50/50 random chance)
- Violations: 0.5-0.7 per step
- Mean SoC: 0.40-0.43

Purpose: Establishes performance floor before training.

### For Trained Policy (After Training)

Target results:
- Improvement rate: >50% (preferably >65%)
- Responsiveness ratio: >0.65 (preferably >0.75)
- Violations: <0.3 per step
- Mean SoC: 0.48-0.52
- Net loading change: negative (reducing congestion)

### Red Flags to Watch For

1. **Agent not learning:**
   - Responsiveness ratio < 0.55
   - Improvement rate < 40%
   - → Check reward balance, train longer, or adjust hyperparameters

2. **BESS unit not working:**
   - Any unit shows "NO ACTIVITY DETECTED"
   - → Check env_helpers.py for indentation errors

3. **Reward imbalance:**
   - Ratio > 10 or < 2
   - → Adjust bonus_constant or soc_penalty_weight

4. **High violations:**
   - Violations > 0.5 per step with trained agent
   - → Increase soc_penalty_weight or flexibility_bonus_weight

## Output Files

Each test run creates:

```
diagnostic_tests/results/
  results_trained_20251028_143055.json   # Summary data
  results_random_20251028_142830.json    # Baseline data
```

JSON files contain:
- All configuration parameters
- Summary metrics
- Episode-by-episode data
- Timestamp and duration

You can compare these files to track improvement over multiple training runs.

## Comparing Baseline vs Trained

Run both tests and compare:

```bash
# 1. Baseline (before training)
python policy_diagnostic_test.py --random

# 2. Train your agent
cd ..
python training.py  # or whatever your training script is

# 3. Evaluate trained agent
cd diagnostic_tests
python policy_diagnostic_test.py --model ../final_model.zip
```

Then compare the key metrics:
- Improvement rate: 0% → 65% ✓
- Responsiveness: 0.50 → 0.72 ✓
- Violations: 0.69 → 0.28 ✓
- Mean SoC: 0.41 → 0.49 ✓

## Impact on Your Simulation

**NONE.** This folder is completely isolated:

- No changes to your original code
- Tests run in separate environment instances
- Results saved to diagnostic_tests/results/
- Does not interfere with training or wandb logging

You can safely run these tests anytime without affecting your main work.

## Quick Reference Commands

```bash
# Navigate to diagnostic tests folder
cd "D:\Thesis\ThesisEnv_refactor\diagnostic_tests"

# Test trained policy (10 episodes, 50 steps each)
python policy_diagnostic_test.py --model ../final_model.zip

# Test random baseline
python policy_diagnostic_test.py --random

# Extended test (20 episodes)
python policy_diagnostic_test.py --model ../final_model.zip --episodes 20

# Get help and see all options
python policy_diagnostic_test.py --help
```

## Questions?

The diagnostic test will answer:
1. ✓ Is congestion reducing? (loading before vs after)
2. ✓ Is agent learning or blind? (responsiveness ratio)
3. ✓ Are all 5 BESS working? (per-unit activity)
4. ✓ Are rewards accurate? (component breakdown)
5. ✓ SoC boundary violations? (upper/lower tracking)

All answered with detailed statistics and episode-by-episode breakdown!
