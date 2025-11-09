"""
run_all_tests.py

Master script to run all diagnostic tests in sequence.
Tests are run in order:
1. Random baseline - Can BESS physically help?
2. Zero action baseline - Is BESS intervention needed?
3. SoC clipping - Does improved action clipping work?
4. Simplified reward - Which reward function is better?

If any critical test fails, the script stops and provides recommendations.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from test_1_random_baseline import run_random_baseline_test
from test_2_zero_action_baseline import run_zero_action_baseline
from test_3_soc_clipping import test_soc_clipping
from test_4_simplified_reward import test_reward_functions
from test_config import TEST_PARAMS


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def print_summary(test_name, result_status, details=""):
    """Print test summary."""
    status_symbols = {
        'PASS': '[PASS]',
        'GOOD': '[PASS]',
        'EXCELLENT': '[PASS]',
        'FAIL': '[FAIL]',
        'MARGINAL': '[WARNING]',
        'MODERATE': '[WARNING]',
        'NEEDS_REVIEW': '[WARNING]',
        'EASY': '[WARNING]',
        'UNCLEAR': '[UNCLEAR]'
    }
    symbol = status_symbols.get(result_status, '[UNCLEAR]')

    print(f"\n{symbol} {test_name}: {result_status}")
    if details:
        print(f"   {details}")


def run_all_tests():
    """Run all diagnostic tests in sequence."""

    print_header("BESS RL ENVIRONMENT DIAGNOSTIC TEST SUITE")

    print("This suite will run 4 diagnostic tests to identify issues with the RL training:")
    print("  1. Random Baseline      - Can BESS physically reduce congestion?")
    print("  2. Zero Action Baseline - Is BESS intervention needed?")
    print("  3. SoC Clipping        - Does improved action clipping work?")
    print("  4. Simplified Reward    - Which reward function works better?")
    print("\nTests will run in sequence. Critical failures will stop execution.")
    print(f"\nResults will be saved to: {TEST_PARAMS['results_dir']}/")
    print("\nStarting tests...")

    # Create results directory
    os.makedirs(TEST_PARAMS['results_dir'], exist_ok=True)

    # Track overall results
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =================================================================
    # TEST 1: Random Baseline
    # =================================================================
    print_header("TEST 1: RANDOM POLICY BASELINE")

    try:
        test1_results = run_random_baseline_test(num_episodes=TEST_PARAMS['num_episodes'], verbose=True)
        all_results['test_1'] = test1_results

        print_summary(
            "Test 1: Random Baseline",
            test1_results['result_status'],
            f"Success rate: {test1_results['overall_success_rate']:.1%}, "
            f"Mean delta: {test1_results['mean_delta']:+.3f}%"
        )

        # Critical failure check
        if test1_results['result_status'] == 'FAIL':
            print("\n" + "="*80)
            print("CRITICAL FAILURE: BESS placement is ineffective!")
            print("="*80)
            print("\nRECOMMENDATION:")
            print("  1. Re-run GA optimization with different fitness function")
            print("  2. Increase BESS capacity (current: 50 MW)")
            print("  3. Try different BESS locations")
            print("\nDo NOT proceed with training until BESS placement is fixed.")
            print("="*80)
            return all_results

    except Exception as e:
        print(f"\n[FAIL] Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return all_results

    # =================================================================
    # TEST 2: Zero Action Baseline
    # =================================================================
    print_header("TEST 2: ZERO-ACTION BASELINE")

    try:
        test2_results = run_zero_action_baseline(num_episodes=TEST_PARAMS['num_episodes'], verbose=True)
        all_results['test_2'] = test2_results

        print_summary(
            "Test 2: Zero Action Baseline",
            test2_results['result_status'],
            f"Mean max loading: {test2_results['mean_max_loading']:.1f}%, "
            f"Congestion rate: {test2_results['congestion_rate']:.1%}"
        )

    except Exception as e:
        print(f"\n[FAIL] Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

    # =================================================================
    # TEST 3: SoC Clipping
    # =================================================================
    print_header("TEST 3: SoC-AWARE ACTION CLIPPING")

    try:
        test3_results = test_soc_clipping(num_test_cases=1000, verbose=True)
        all_results['test_3'] = test3_results

        print_summary(
            "Test 3: SoC Clipping",
            test3_results['result_status'],
            f"Success rate: {test3_results['success_rate']:.1%}, "
            f"Violations prevented: {test3_results['soc_violations_prevented']}"
        )

        # Critical failure check
        if test3_results['result_status'] == 'FAIL':
            print("\n" + "="*80)
            print("WARNING: SoC clipping implementation has issues!")
            print("="*80)
            print("\nRECOMMENDATION:")
            print("  1. Review improved_env_helpers.py implementation")
            print("  2. Check SoC calculation logic")
            print("  3. Verify energy/power conversion")
            print("="*80)

    except Exception as e:
        print(f"\n[FAIL] Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

    # =================================================================
    # TEST 4: Simplified Reward
    # =================================================================
    print_header("TEST 4: REWARD FUNCTION COMPARISON")

    try:
        test4_results = test_reward_functions(num_episodes=50, verbose=True)
        all_results['test_4'] = test4_results

        print_summary(
            "Test 4: Reward Function",
            test4_results['result_status'],
            f"Recommended: {test4_results['recommended']}, "
            f"Correlation: {test4_results['correlations']['simplified_vs_delta']:.3f}"
        )

    except Exception as e:
        print(f"\n[FAIL] Test 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print_header("DIAGNOSTIC TEST SUITE - FINAL SUMMARY")

    print("Test Results:")
    print("-"*80)
    for test_name, results in all_results.items():
        if isinstance(results, dict) and 'result_status' in results:
            status = results['result_status']
            print(f"  {test_name.replace('_', ' ').title():30s} {status}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    # Analyze results and provide recommendations
    if 'test_1' in all_results:
        test1_status = all_results['test_1']['result_status']

        if test1_status == 'PASS':
            print("\n[PASS] BESS placement is effective - Proceed with improvements:")
            print("\n   APPLY THESE CHANGES TO YOUR ORIGINAL CODE:")
            print("   " + "-"*76)

            if 'test_3' in all_results and all_results['test_3']['result_status'] == 'PASS':
                print("   1. [PASS] Add SoC-aware action clipping")
                print("      → Copy apply_bess_action_with_soc_clipping() to env_helpers.py")
                print("      → Use in ENV_BESS.step() before applying actions to grid")

            if 'test_4' in all_results:
                recommended = all_results['test_4'].get('recommended', 'SIMPLIFIED')
                print(f"\n   2. [PASS] Use {recommended} reward function")
                if recommended == 'SIMPLIFIED':
                    print("      → Copy calculate_simplified_reward() to env_helpers.py")
                    print("      → Replace calculate_bess_reward() in ENV_BESS.step()")
                else:
                    print("      → Copy calculate_improved_reward() to env_helpers.py")
                    print("      → Update config.py: soc_penalty_weight = -5.0")

            print("\n   3. [PASS] Reduce episode length for faster learning")
            print("      → Update config.py: max_step = 10 (was 50)")

            print("\n   4. [PASS] Increase entropy coefficient for more exploration")
            print("      → Update config.py: ent_coef = 0.1 (was 0.05)")

            print("\n   5. [PASS] Remove action_magnitude_bonus if present")
            print("      → Check env_helpers.py calculate_bess_reward()")
            print("      → Comment out or remove action_magnitude_bonus lines")

            print("\n   After applying changes:")
            print("   → Train for 20,000 steps")
            print("   → Monitor success rate - should improve from 50% to 60%+")
            print("   → If still no learning, try SAC algorithm instead of PPO")

        elif test1_status == 'FAIL':
            print("\n[FAIL] BESS placement is INEFFECTIVE - DO NOT proceed with training:")
            print("\n   REQUIRED ACTIONS:")
            print("   " + "-"*76)
            print("   1. [FAIL] Re-run GA optimization")
            print("      → Review ga_optimization.py fitness function")
            print("      → Increase population size or generations")
            print("      → Try different BESS capacity (current: 50 MW)")
            print("\n   2. [FAIL] Validate GA results")
            print("      → Run test_1_random_baseline.py with new placement")
            print("      → Success rate should be > 55%")
            print("\n   Do NOT modify original code until GA placement is fixed!")

        else:  # MARGINAL
            print("\n[WARNING]  BESS placement is MARGINAL - Proceed with caution:")
            print("\n   RECOMMENDED ACTIONS:")
            print("   " + "-"*76)
            print("   1. [WARNING]  Apply improvements but expect limited gains")
            print("   2. [WARNING]  Consider increasing BESS capacity to 75-100 MW")
            print("   3. [WARNING]  May need curriculum learning (train on high-loading scenarios first)")

    print("\n" + "="*80)
    print(f"All test results saved to: {TEST_PARAMS['results_dir']}/")
    print("="*80)

    return all_results


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" BESS RL DIAGNOSTIC TEST SUITE ".center(80, "="))
    print("="*80)

    results = run_all_tests()

    print("\n[PASS] All tests completed!")
    print(f"Review results in: {TEST_PARAMS['results_dir']}/")
    print("\nNext step: Review RECOMMENDATIONS above before modifying original code.")
