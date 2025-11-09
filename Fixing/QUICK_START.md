# Quick Start Guide - Diagnostic Tests

## What This Folder Contains

This `Fixing/` folder contains diagnostic tests to identify and fix the RL training issues. **DO NOT modify your original code yet!**

All tests run in this isolated folder and will tell you exactly what changes to make if they pass.

---

## Quick Test (Recommended First)

Run this quick 5-episode test to see if BESS can reduce congestion:

```bash
cd Fixing
python -c "
import sys
sys.path.insert(0, '..')
from ENV_BESS_main import ENV_BESS
from test_config import get_test_env_config
import numpy as np

env = ENV_BESS(**get_test_env_config())
successes, deltas = [], []

for ep in range(5):
    obs, info = env.reset()
    for step in range(10):
        obs, reward, done, trunc, info = env.step(np.random.uniform(-1, 1, 5))
        try:
            delta = env.loading_before_action - env.loading_after_action
            deltas.append(delta)
            successes.append(1 if delta > 0 else 0)
        except: pass
        if done or trunc: break

success_rate = sum(successes) / len(successes) if successes else 0
print(f'\nSuccess Rate: {success_rate:.1%}')
print(f'Mean Delta: {sum(deltas)/len(deltas):+.3f}%' if deltas else 'No data')
print('PASS - Proceed' if success_rate > 0.55 else 'FAIL - Check BESS placement')
"
```

**Expected output:**
- Success Rate: 50-60%
- Mean Delta: ~0%
- If < 50%: BESS placement is bad, re-run GA optimization

---

## Full Test Suite

If quick test passes, run all tests:

```bash
cd Fixing
python run_all_tests.py
```

This will run:
1. Random Baseline (20 episodes) - ~5 minutes
2. Zero Action Baseline (20 episodes) - ~5 minutes
3. SoC Clipping Test (1000 scenarios) - ~2 minutes
4. Reward Function Comparison (50 episodes) - ~10 minutes

**Total time: ~25 minutes**

---

## What To Do After Tests Complete

### If Test 1 PASSES (success rate > 55%):

✅ **Good news!** Your BESS placement works. Apply these changes to original code:

1. **Add SoC clipping** (prevents impossible actions):
   - Copy `apply_bess_action_with_soc_clipping()` from `improved_env_helpers.py` → `env_helpers.py`
   - Update `ENV_BESS.step()` to use it

2. **Simplify reward function** (if test 4 recommends it):
   - Copy `calculate_simplified_reward()` from `improved_env_helpers.py` → `env_helpers.py`
   - Replace `calculate_bess_reward()` in `ENV_BESS.step()`

3. **Update config.py**:
   ```python
   'max_step': 10,  # was 50
   'soc_penalty_weight': -5.0,  # was -0.2
   'ent_coef': 0.15,  # was 0.10
   ```

4. **Train for 20k steps** and check if success rate increases from 50% to 60%+

See `CHANGES_TO_APPLY.md` for detailed instructions.

---

### If Test 1 FAILS (success rate < 48%):

❌ **Problem:** BESS placement is ineffective. Do NOT modify original code yet!

**Required actions:**
1. Re-run GA optimization with different parameters
2. Try increasing BESS capacity from 50 MW to 75-100 MW
3. Validate new placement with Test 1 again
4. Only proceed when success rate > 55%

---

## Test Results Location

All results saved to: `Fixing/results/`

Each test creates a JSON file with timestamp:
- `test_1_random_baseline_YYYYMMDD_HHMMSS.json`
- `test_2_zero_action_YYYYMMDD_HHMMSS.json`
- `test_3_soc_clipping_YYYYMMDD_HHMMSS.json`
- `test_4_reward_comparison_YYYYMMDD_HHMMSS.json`

---

## Troubleshooting

### Tests fail with import errors:
```bash
# Make sure you're in the Fixing directory
cd Fixing
python run_all_tests.py
```

### Tests run very slowly:
Edit `test_config.py`:
```python
TEST_PARAMS = {
    'num_episodes': 10,  # Reduce from 20 to 10
    # ...
}
```

### Need to re-run specific test:
```bash
cd Fixing
python test_1_random_baseline.py  # Just test 1
python test_3_soc_clipping.py     # Just test 3
```

---

## Important Reminders

⚠️ **DO NOT modify original code until tests pass!**

⚠️ **DO NOT skip Test 1** - it validates BESS placement

✅ **DO review** `CHANGES_TO_APPLY.md` before applying fixes

✅ **DO backup** your original files before making changes

---

## Quick Decision Tree

```
Run Quick Test (5 episodes)
│
├─ Success rate > 55%
│  └─ ✅ Run full test suite
│     └─ Apply improvements from CHANGES_TO_APPLY.md
│
├─ Success rate 45-55%
│  └─ ⚠️  MARGINAL - May need BESS capacity increase
│     └─ Run full tests, apply changes, monitor carefully
│
└─ Success rate < 45%
   └─ ❌ FAIL - Re-run GA optimization
      └─ DO NOT proceed until fixed
```

---

## Questions?

1. Read `README.md` for detailed test descriptions
2. Read `CHANGES_TO_APPLY.md` for step-by-step change instructions
3. Check test results in `results/` directory
4. Review improved functions in `improved_env_helpers.py`

**Remember: This folder is for testing only. Original code stays unchanged until tests confirm improvements work!**
