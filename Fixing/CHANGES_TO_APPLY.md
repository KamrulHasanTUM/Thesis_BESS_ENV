# Changes to Apply to Original Code

**IMPORTANT: Only apply these changes AFTER all diagnostic tests pass!**

---

## Prerequisites

Run the diagnostic tests first:
```bash
cd Fixing
python run_all_tests.py
```

Wait for all tests to complete and review the recommendations.

---

## Changes to Apply (If Tests Pass)

### 1. Update `env_helpers.py` - Add SoC-Aware Action Clipping

**Location:** `env_helpers.py` (add new function before `apply_bess_action`)

**Action:** Copy the entire `apply_bess_action_with_soc_clipping()` function from `Fixing/improved_env_helpers.py`

```python
# Add this function to env_helpers.py

def apply_bess_action_with_soc_clipping(env, action_normalized):
    """
    Apply BESS action with SoC-aware clipping to prevent impossible operations.

    [Copy full implementation from Fixing/improved_env_helpers.py]
    """
    # ... (full implementation)
```

---

### 2. Update `ENV_BESS_main.py` - Use SoC Clipping in step()

**Location:** `ENV_BESS_main.py`, in the `step()` method

**Before:**
```python
def step(self, action):
    # ... existing code ...

    # Apply BESS action
    helpers.apply_bess_action(self, action)
```

**After:**
```python
def step(self, action):
    # ... existing code ...

    # Apply BESS action with SoC-aware clipping
    clipped_action, clip_info = helpers.apply_bess_action_with_soc_clipping(self, action)

    # Optional: Log clipping events
    if len(clip_info['units_clipped']) > 0:
        print(f"Warning: {len(clip_info['units_clipped'])} BESS units clipped")
        for reason in clip_info['clip_reasons']:
            print(f"  {reason}")

    # Now apply the clipped action to grid
    helpers.apply_bess_action(self, clipped_action / self.bess_power_mw)  # Renormalize
```

---

### 3. Update `env_helpers.py` - Replace Reward Function

**Option A: Use Simplified Reward (if test_4 recommends "SIMPLIFIED")**

**Location:** `env_helpers.py`, in `calculate_bess_reward()` function

**Replace entire function with:**
```python
def calculate_bess_reward(env, max_loading_before, max_loading_after):
    """
    Simplified reward function - ONLY congestion reduction.
    Removes all other components to provide clean learning signal.
    """
    # Pure congestion reduction
    delta_loading = max_loading_before - max_loading_after

    # Scale to reasonable range
    congestion_reward = delta_loading * 10.0

    # Clip to prevent extreme values
    congestion_reward = np.clip(congestion_reward, -500, 500)

    reward_breakdown = {
        'congestion': float(congestion_reward),
        'delta_loading': float(delta_loading)
    }

    return congestion_reward, reward_breakdown
```

**Option B: Use Improved Reward (if test_4 recommends "IMPROVED")**

**Location:** `env_helpers.py`, in `calculate_bess_reward()` function

1. **Remove** action_magnitude_bonus section (lines 1103-1105 and 1085-1101):
   ```python
   # DELETE THESE LINES:
   # action_magnitude_weight = 100.0
   # avg_power_utilization = np.mean(np.abs(env.bess_power)) / env.bess_power_mw
   # action_magnitude_bonus = action_magnitude_weight * avg_power_utilization
   ```

2. **Update** total_reward calculation (line 1108):
   ```python
   # OLD:
   total_reward = congestion_reward + soc_penalty + action_magnitude_bonus

   # NEW:
   total_reward = congestion_reward + soc_penalty
   ```

3. **Update** reward_breakdown (line 1111-1115):
   ```python
   # OLD:
   reward_breakdown = {
       'congestion': float(congestion_reward),
       'soc_penalty': float(soc_penalty),
       'action_magnitude_bonus': float(action_magnitude_bonus),
   }

   # NEW:
   reward_breakdown = {
       'congestion': float(congestion_reward),
       'soc_penalty': float(soc_penalty),
   }
   ```

---

### 4. Update `config.py` - Increase SoC Penalty

**Location:** `config.py`, line 110

**Before:**
```python
'soc_penalty_weight': -0.2  # Reduced from -1.0 to allow more freedom
```

**After:**
```python
'soc_penalty_weight': -5.0  # Increased to strongly discourage boundary violations
```

---

### 5. Update `config.py` - Reduce Episode Length

**Location:** `config.py`, line 52

**Before:**
```python
'max_step': 50,
```

**After:**
```python
'max_step': 10,  # Reduced for faster learning and credit assignment
```

---

### 6. Update `config.py` - Increase Entropy Coefficient

**Location:** `config.py`, line 124

**Before:**
```python
'ent_coef': 0.10,
```

**After:**
```python
'ent_coef': 0.15,  # Increased for more exploration
```

---

### 7. Update `config.py` - Adjust Bonus Constant (Optional)

**Location:** `config.py`, line 58

**Before:**
```python
'bonus_constant': 100,  # Reduced from 50000 to make rewards learnable
```

**After (if using improved reward):**
```python
'bonus_constant': 100,  # Keep at 100 - works well with improved reward
```

**After (if using simplified reward):**
```python
# This parameter is not used in simplified reward, but can leave as is
'bonus_constant': 100,
```

---

## Testing the Changes

After applying changes:

1. **Quick sanity check:**
   ```bash
   python ENV_BESS_main.py
   ```
   Verify no import errors or syntax issues.

2. **Short training run:**
   ```bash
   # Update init_meta.json:
   {
       "exp_code": "improved_v1",
       "exp_id": 2,
       "exp_name": "with_soc_clipping_and_simplified_reward",
       "grid_env": "bess"
   }

   # Run training
   python ENV_BESS_main.py
   ```

3. **Monitor training (first 5,000 steps):**
   - Check W&B dashboard
   - Look for:
     - ✅ Success rate increasing from 50% to 55-60%
     - ✅ Episode rewards improving (less negative)
     - ✅ SoC violations decreasing
     - ✅ Power utilization remaining 30-40%

4. **If success rate still ~50% after 10k steps:**
   - Try switching to SAC algorithm (see Advanced Changes below)
   - Or increase BESS capacity to 75-100 MW

---

## Advanced Changes (If Basic Changes Don't Work)

### Switch from PPO to SAC

**Location:** `training.py`, in `create_model()` function

**Before:**
```python
from stable_baselines3 import PPO

def create_model(env, training_config, logdir):
    # ...
    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        # ... other params
    )
```

**After:**
```python
from stable_baselines3 import SAC

def create_model(env, training_config, logdir):
    # ...
    return SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,  # SAC typically uses 3e-4
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',  # Automatic entropy tuning
        tensorboard_log=logdir,
        verbose=1
    )
```

---

## Verification Checklist

Before running long training:

- [ ] Test 1 (Random Baseline) passed with success rate > 55%
- [ ] Test 3 (SoC Clipping) passed with success rate > 99%
- [ ] SoC clipping function added to env_helpers.py
- [ ] ENV_BESS.step() updated to use SoC clipping
- [ ] Reward function updated (simplified or improved)
- [ ] config.py updated with new parameters
- [ ] Short test run (100 steps) completes without errors
- [ ] W&B logging still working

---

## Expected Results After Changes

**Within 10,000 training steps:**
- Success rate: 50% → 60-65%
- Episode rewards: More positive episodes
- SoC violations: Reduced by 80%+
- Agent behavior: More strategic (scales actions to congestion)

**Within 30,000 training steps:**
- Success rate: 65-70%
- Clear learning trend (upward slope)
- Max loading consistently reduced

**If no improvement after 20,000 steps:**
- Re-run Test 1 (Random Baseline) to verify BESS can help
- Try SAC algorithm
- Consider curriculum learning (train on high-loading scenarios only)

---

## Rollback Instructions

If changes cause issues:

1. **Backup first:** (before applying changes)
   ```bash
   cp env_helpers.py env_helpers.py.backup
   cp ENV_BESS_main.py ENV_BESS_main.py.backup
   cp config.py config.py.backup
   ```

2. **Restore if needed:**
   ```bash
   cp env_helpers.py.backup env_helpers.py
   cp ENV_BESS_main.py.backup ENV_BESS_main.py
   cp config.py.backup config.py
   ```

---

## Questions?

If unclear about any changes:
1. Review the test results in `Fixing/results/`
2. Check the recommendations in the test output
3. Compare `Fixing/improved_env_helpers.py` with original `env_helpers.py`

**Remember: Only apply changes after tests pass and recommendations are clear!**
