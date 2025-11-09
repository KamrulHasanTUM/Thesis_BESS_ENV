"""
Final implementation verification - checks all improvements are working.
This is a clean test without any cache issues.
"""

import numpy as np
from ENV_BESS_main import ENV_BESS
from config import load_config, create_bess_env_config, create_training_config

def verify_implementation():
    """Verify all improvements from the report."""
    print("=" * 80)
    print("FINAL IMPLEMENTATION VERIFICATION")
    print("=" * 80)
    print()

    # Load configuration
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)
    training_config = create_training_config(init_meta)

    print("STEP 1: Verify PPO Hyperparameters (Section 12.6)")
    print("-" * 80)
    print(f"clip_range: {training_config['clip_range']} (expected: 0.3)")
    print(f"total_timesteps: {training_config['total_timesteps']} (expected: 150000)")
    print(f"initial_learning_rate: {training_config['initial_learning_rate']} (expected: 0.0003)")

    hyperparams_ok = (
        training_config['clip_range'] == 0.3 and
        training_config['total_timesteps'] == 150000 and
        training_config['initial_learning_rate'] == 0.0003
    )
    print(f"Status: {'[OK]' if hyperparams_ok else '[FAIL]'}")
    print()

    # Create environment
    print("STEP 2: Create Environment")
    print("-" * 80)
    env = ENV_BESS(**env_config)
    print("[OK] Environment created")
    print()

    # Check observation space
    print("STEP 3: Verify Feature Selection (Section 12.7)")
    print("-" * 80)
    obs_space_keys = list(env.observation_space.spaces.keys())
    print(f"Number of observation features: {len(obs_space_keys)}")
    print(f"Feature names: {sorted(obs_space_keys)}")
    print()

    # Check removed features
    has_discrete_switches = 'discrete_switches' in obs_space_keys
    has_vm_bus = 'continuous_vm_bus' in obs_space_keys

    print("Checking removed features:")
    print(f"  discrete_switches: {'[FAIL - STILL PRESENT]' if has_discrete_switches else '[OK - REMOVED]'}")
    print(f"  continuous_vm_bus: {'[FAIL - STILL PRESENT]' if has_vm_bus else '[OK - REMOVED]'}")
    print()

    features_removed_ok = not has_discrete_switches and not has_vm_bus

    # Reset and check observation
    print("STEP 4: Verify Observation Space Enhancements (Section 12.1-12.3)")
    print("-" * 80)
    observation, info = env.reset()
    print(f"Observation keys in actual observation: {len(observation.keys())}")
    print()

    # Check new features
    has_top_k = 'top_k_congested_lines' in observation
    has_sensitivity = 'bess_sensitivity_to_top_k' in observation
    has_trend = 'loading_trend' in observation

    print("Checking new features:")
    print(f"  top_k_congested_lines: {'[OK]' if has_top_k else '[FAIL]'}")
    if has_top_k:
        print(f"    Shape: {observation['top_k_congested_lines'].shape} (expected: (10,))")

    print(f"  bess_sensitivity_to_top_k: {'[OK]' if has_sensitivity else '[FAIL]'}")
    if has_sensitivity:
        print(f"    Shape: {observation['bess_sensitivity_to_top_k'].shape} (expected: (5, 10))")

    print(f"  loading_trend: {'[OK]' if has_trend else '[FAIL]'}")
    if has_trend:
        print(f"    Shape: {observation['loading_trend'].shape} (expected: (95,))")

    print()
    enhancements_ok = has_top_k and has_sensitivity and has_trend

    # Take a step to verify sensitivity computation
    print("STEP 5: Verify Dynamic Feature Updates")
    print("-" * 80)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    sens_nonzero = np.count_nonzero(observation['bess_sensitivity_to_top_k'])
    print(f"Sensitivity non-zero count after step: {sens_nonzero}/50")
    print(f"Sensitivity computation: {'[OK]' if sens_nonzero > 0 else '[WARN - all zeros]'}")
    print()

    # Take another step for trend
    observation, reward, terminated, truncated, info = env.step(action)
    trend_nonzero = np.count_nonzero(observation['loading_trend'])
    print(f"Loading trend non-zero count after 2nd step: {trend_nonzero}/95")
    print(f"Trend computation: {'[OK]' if trend_nonzero > 0 else '[WARN - all zeros]'}")
    print()

    # Count total features
    print("STEP 6: Dimensionality Analysis")
    print("-" * 80)

    total_features = 0
    for key, value in observation.items():
        if hasattr(value, 'shape'):
            if len(value.shape) == 1:
                count = value.shape[0]
            elif len(value.shape) == 2:
                count = value.shape[0] * value.shape[1]
            else:
                count = np.prod(value.shape)
            total_features += count
            print(f"  {key:35s}: {str(value.shape):15s} = {count:4d} features")

    print()
    print(f"TOTAL FEATURES: {total_features}")
    print(f"Expected: ~315 features (67% reduction from original ~960)")
    print(f"Status: {'[OK]' if 300 <= total_features <= 350 else '[CHECK]'}")
    print()

    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    all_checks = [
        ("PPO Hyperparameters (12.6)", hyperparams_ok),
        ("Feature Selection (12.7)", features_removed_ok),
        ("Observation Enhancements (12.1-12.3)", enhancements_ok),
        ("Sensitivity Computation", sens_nonzero > 0),
        ("Trend Computation", trend_nonzero > 0),
    ]

    passed = sum(1 for _, ok in all_checks if ok)
    total = len(all_checks)

    for name, status in all_checks:
        print(f"  {name:40s}: {'[OK]' if status else '[FAIL]'}")

    print()
    print(f"TESTS PASSED: {passed}/{total}")
    print()

    if passed == total:
        print("=" * 80)
        print("*** ALL IMPLEMENTATIONS VERIFIED! READY FOR 150K TRAINING! ***")
        print("=" * 80)
        print()
        print("To start training:")
        print("  python ENV_BESS_main.py")
        return True
    else:
        print("=" * 80)
        print("*** SOME CHECKS FAILED - PLEASE REVIEW ***")
        print("=" * 80)
        return False


if __name__ == "__main__":
    try:
        success = verify_implementation()
        exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 80)
        print("*** VERIFICATION FAILED WITH EXCEPTION ***")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
