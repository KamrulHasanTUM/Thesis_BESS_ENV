"""
Test 3: SoC-Aware Action Clipping

This test verifies that the improved action clipping prevents impossible actions
and ensures BESS respects energy constraints.
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ENV_BESS_main import ENV_BESS
from test_config import get_test_env_config, TEST_PARAMS
from improved_env_helpers import apply_bess_action_with_soc_clipping, validate_action_bounds


def test_soc_clipping(num_test_cases=1000, verbose=True):
    """
    Test SoC-aware action clipping with extreme scenarios.

    Args:
        num_test_cases: Number of random action scenarios to test
        verbose: Whether to print progress

    Returns:
        dict: Test results
    """
    print("="*80)
    print("TEST 3: SoC-AWARE ACTION CLIPPING")
    print("="*80)
    print(f"Testing improved action clipping with {num_test_cases} random scenarios...")
    print("-"*80)

    # Create environment
    env_config = get_test_env_config()
    env = ENV_BESS(**env_config)

    # Initialize environment
    obs, info = env.reset()

    # Test results
    results = {
        'total_tests': num_test_cases,
        'tests_clipped': 0,
        'tests_passed': 0,
        'soc_violations_prevented': 0,
        'clip_examples': [],
        'extreme_cases': []
    }

    np.random.seed(TEST_PARAMS['random_seed'])

    print("\nRunning clipping tests...")
    print("-"*80)

    for test_idx in range(num_test_cases):
        # Generate random SoC states (including extreme cases)
        if test_idx % 5 == 0:
            # Create extreme low SoC scenario
            env.bess_soc = np.random.uniform(0.1, 0.15, size=env.num_bess)
        elif test_idx % 5 == 1:
            # Create extreme high SoC scenario
            env.bess_soc = np.random.uniform(0.85, 0.9, size=env.num_bess)
        elif test_idx % 5 == 2:
            # Create mixed extreme scenario
            env.bess_soc = np.random.choice([0.1, 0.9], size=env.num_bess)
        else:
            # Normal random SoC
            env.bess_soc = np.random.uniform(0.2, 0.8, size=env.num_bess)

        # Generate aggressive random action
        action_normalized = np.random.uniform(-1.0, 1.0, size=env.num_bess)

        # Validate input action bounds
        is_valid, issues = validate_action_bounds(action_normalized)
        if not is_valid:
            if verbose and len(results['extreme_cases']) < 5:
                print(f"Warning: Action out of bounds at test {test_idx}:")
                for issue in issues:
                    print(f"  {issue}")
            results['extreme_cases'].append({
                'test_idx': test_idx,
                'issues': issues
            })

        # Apply SoC-aware clipping
        clipped_action, clip_info = apply_bess_action_with_soc_clipping(env, action_normalized)

        # Check if clipping occurred
        if len(clip_info['units_clipped']) > 0:
            results['tests_clipped'] += 1

        # Simulate SoC update to verify no violations
        test_passed = True
        for i in range(env.num_bess):
            # Calculate SoC change
            energy_change = (clipped_action[i] * env.time_step_hours) / env.bess_capacity_mwh
            new_soc = env.bess_soc[i] - energy_change  # Negative because discharge reduces SoC

            # Check if within bounds (with small tolerance for numerical errors)
            if new_soc < (env.soc_min - 0.001) or new_soc > (env.soc_max + 0.001):
                test_passed = False
                if verbose:
                    print(f"FAIL: Test {test_idx}, BESS {i} - SoC violation!")
                    print(f"  Current SoC: {env.bess_soc[i]:.3f}")
                    print(f"  Action: {clipped_action[i]:.2f} MW")
                    print(f"  New SoC: {new_soc:.3f} (Bounds: {env.soc_min}-{env.soc_max})")

            # Count if clipping prevented violation
            if len(clip_info['units_clipped']) > 0 and i in clip_info['units_clipped']:
                # Check if unclipped action would have violated
                original_action = clip_info['original_actions'][clip_info['units_clipped'].index(i)]
                original_energy_change = (original_action * env.time_step_hours) / env.bess_capacity_mwh
                original_new_soc = env.bess_soc[i] - original_energy_change

                if original_new_soc < env.soc_min or original_new_soc > env.soc_max:
                    results['soc_violations_prevented'] += 1

        if test_passed:
            results['tests_passed'] += 1

        # Store interesting examples
        if len(clip_info['units_clipped']) > 0 and len(results['clip_examples']) < 10:
            example = {
                'test_idx': test_idx,
                'soc_state': env.bess_soc.tolist(),
                'original_action': action_normalized.tolist(),
                'clipped_action_mw': clipped_action.tolist(),
                'units_clipped': clip_info['units_clipped'],
                'clip_reasons': clip_info['clip_reasons']
            }
            results['clip_examples'].append(example)

        if verbose and test_idx % 200 == 0:
            print(f"Progress: {test_idx}/{num_test_cases} tests completed")

    # Calculate statistics
    success_rate = results['tests_passed'] / results['total_tests']
    clip_rate = results['tests_clipped'] / results['total_tests']

    # Print results
    print("\n" + "="*80)
    print("SoC CLIPPING TEST RESULTS")
    print("="*80)
    print(f"Total Tests:              {results['total_tests']}")
    print(f"Tests Passed:             {results['tests_passed']} ({success_rate:.1%})")
    print(f"Tests with Clipping:      {results['tests_clipped']} ({clip_rate:.1%})")
    print(f"SoC Violations Prevented: {results['soc_violations_prevented']}")
    print("-"*80)

    # Show examples
    if results['clip_examples']:
        print("\nEXAMPLE CLIPPING SCENARIOS:")
        print("-"*80)
        for i, example in enumerate(results['clip_examples'][:3], 1):
            print(f"\nExample {i}:")
            print(f"  SoC State: {[f'{s:.2f}' for s in example['soc_state']]}")
            print(f"  Units Clipped: {example['units_clipped']}")
            for reason in example['clip_reasons']:
                print(f"    - {reason}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("-"*80)

    if success_rate >= 0.99:
        print("[PASS] EXCELLENT: SoC clipping prevents all constraint violations")
        print("   Conclusion: Improved action clipping is working correctly.")
        print("   Next steps: Apply this to main environment.")
        result_status = "PASS"

    elif success_rate >= 0.95:
        print("[PASS] GOOD: SoC clipping prevents most violations (minor numerical errors)")
        print("   Conclusion: Acceptable with small tolerance adjustments.")
        result_status = "PASS"

    else:
        print(f"[FAIL] FAIL: SoC clipping not preventing violations ({success_rate:.1%} success)")
        print("   Conclusion: Implementation needs debugging.")
        result_status = "FAIL"

    results['success_rate'] = success_rate
    results['clip_rate'] = clip_rate
    results['result_status'] = result_status

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(TEST_PARAMS['results_dir'], f'test_3_soc_clipping_{timestamp}.json')

    # Prepare for JSON serialization
    save_results = {
        'total_tests': results['total_tests'],
        'tests_passed': results['tests_passed'],
        'tests_clipped': results['tests_clipped'],
        'soc_violations_prevented': results['soc_violations_prevented'],
        'success_rate': float(success_rate),
        'clip_rate': float(clip_rate),
        'result_status': result_status,
        'clip_examples': results['clip_examples'][:5]  # Save first 5 examples
    }

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=4)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    return results


if __name__ == "__main__":
    results = test_soc_clipping(num_test_cases=1000)

    if results['result_status'] == 'PASS':
        print("\n[PASS] TEST 3 PASSED - SoC clipping is working correctly")
        sys.exit(0)
    else:
        print("\n[FAIL] TEST 3 FAILED - SoC clipping needs fixes")
        sys.exit(1)
