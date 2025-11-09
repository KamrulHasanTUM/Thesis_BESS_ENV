"""
Test 1: Random Policy Baseline

This test evaluates whether BESS can physically reduce congestion at the GA-optimized locations.
If random actions can reduce congestion ~50-60% of the time, then the agent should be able to learn.
If random actions only succeed ~50% of the time, BESS placement may be suboptimal.
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


def run_random_baseline_test(num_episodes=100, verbose=True):
    """
    Test if BESS can physically reduce congestion with random actions.

    Args:
        num_episodes: Number of episodes to test
        verbose: Whether to print progress

    Returns:
        dict: Test results with statistics
    """
    print("="*80)
    print("TEST 1: RANDOM POLICY BASELINE")
    print("="*80)
    print(f"Testing if BESS can reduce congestion with random actions...")
    print(f"Episodes: {num_episodes}")
    print(f"BESS Locations: [39, 189, 230, 281, 282]")
    print(f"BESS Power: 50 MW per unit")
    print("-"*80)

    # Create environment
    env_config = get_test_env_config()
    env = ENV_BESS(**env_config)

    # Storage for results
    results = {
        'episode_success_rates': [],
        'episode_mean_deltas': [],
        'all_deltas': [],
        'loading_before_all': [],
        'loading_after_all': [],
        'step_success_count': 0,
        'total_steps': 0
    }

    np.random.seed(TEST_PARAMS['random_seed'])

    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_deltas = []

        for step in range(env.max_step):
            # Random action between -50 and +50 MW
            random_action = np.random.uniform(-1.0, 1.0, size=env.num_bess)

            # Take step
            obs, reward, terminated, truncated, info = env.step(random_action)

            # Get loading metrics
            try:
                loading_before = env.loading_before_action
                loading_after = env.loading_after_action
                delta = loading_before - loading_after

                episode_deltas.append(delta)
                results['all_deltas'].append(delta)
                results['loading_before_all'].append(loading_before)
                results['loading_after_all'].append(loading_after)

                if delta > 0:
                    results['step_success_count'] += 1

                results['total_steps'] += 1

            except AttributeError:
                # Environment might not expose these attributes
                pass

            if terminated or truncated:
                break

        # Episode statistics
        if len(episode_deltas) > 0:
            success_rate = np.sum([d > 0 for d in episode_deltas]) / len(episode_deltas)
            mean_delta = np.mean(episode_deltas)

            results['episode_success_rates'].append(success_rate)
            results['episode_mean_deltas'].append(mean_delta)

            if verbose and episode % 10 == 0:
                print(f"Episode {episode:3d}: Success={success_rate:.1%}, Mean Delta={mean_delta:+.2f}%")

    # Calculate overall statistics
    overall_stats = {
        'overall_success_rate': results['step_success_count'] / results['total_steps'] if results['total_steps'] > 0 else 0,
        'mean_success_rate': np.mean(results['episode_success_rates']) if results['episode_success_rates'] else 0,
        'std_success_rate': np.std(results['episode_success_rates']) if results['episode_success_rates'] else 0,
        'mean_delta': np.mean(results['all_deltas']) if results['all_deltas'] else 0,
        'std_delta': np.std(results['all_deltas']) if results['all_deltas'] else 0,
        'median_delta': np.median(results['all_deltas']) if results['all_deltas'] else 0,
        'best_delta': np.max(results['all_deltas']) if results['all_deltas'] else 0,
        'worst_delta': np.min(results['all_deltas']) if results['all_deltas'] else 0,
        'num_episodes': num_episodes,
        'total_steps': results['total_steps']
    }

    # Print results
    print("\n" + "="*80)
    print("RANDOM BASELINE TEST RESULTS")
    print("="*80)
    print(f"Overall Success Rate:     {overall_stats['overall_success_rate']:.1%}")
    print(f"Mean Success Rate:        {overall_stats['mean_success_rate']:.1%} ± {overall_stats['std_success_rate']:.1%}")
    print(f"Mean Loading Delta:       {overall_stats['mean_delta']:+.3f}%")
    print(f"Std Dev of Delta:         {overall_stats['std_delta']:.3f}%")
    print(f"Median Delta:             {overall_stats['median_delta']:+.3f}%")
    print(f"Best Delta (Reduction):   {overall_stats['best_delta']:+.3f}%")
    print(f"Worst Delta (Increase):   {overall_stats['worst_delta']:+.3f}%")
    print(f"Total Steps Tested:       {overall_stats['total_steps']}")
    print("-"*80)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-"*80)

    success_rate = overall_stats['overall_success_rate']
    mean_delta = overall_stats['mean_delta']

    if success_rate >= 0.55 and mean_delta > 0:
        print("[PASS] GOOD: BESS CAN reduce congestion (success rate > 55%)")
        print("   Conclusion: Agent should be able to learn from this task.")
        print("   Next steps: Proceed with reward function and training improvements.")
        result_status = "PASS"

    elif 0.48 <= success_rate <= 0.52 and abs(mean_delta) < 0.5:
        print("[WARNING] BESS has MINIMAL impact (success ~50%, delta ~0)")
        print("   Conclusion: BESS placement may be suboptimal or capacity insufficient.")
        print("   Next steps: Consider re-running GA optimization or increasing BESS capacity.")
        result_status = "MARGINAL"

    elif success_rate < 0.48 and mean_delta < 0:
        print("[FAIL] BESS WORSENS congestion on average (success < 48%)")
        print("   Conclusion: BESS placement is counterproductive.")
        print("   Next steps: MUST re-run GA optimization with different parameters.")
        result_status = "FAIL"

    else:
        print("[UNCLEAR] Results are mixed or inconclusive.")
        print(f"   Success rate: {success_rate:.1%}, Mean delta: {mean_delta:+.3f}%")
        print("   Next steps: Review BESS placement and capacity settings.")
        result_status = "UNCLEAR"

    overall_stats['result_status'] = result_status

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(TEST_PARAMS['results_dir'], f'test_1_random_baseline_{timestamp}.json')

    os.makedirs(TEST_PARAMS['results_dir'], exist_ok=True)

    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        save_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in overall_stats.items()}
        json.dump(save_stats, f, indent=4)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    return overall_stats


if __name__ == "__main__":
    results = run_random_baseline_test(num_episodes=TEST_PARAMS['num_episodes'])

    # Exit with appropriate code
    if results['result_status'] == 'PASS':
        print("\n[PASS] TEST 1 PASSED - Proceed to Test 2")
        sys.exit(0)
    elif results['result_status'] == 'MARGINAL':
        print("\n[WARNING] TEST 1 MARGINAL - Review results before proceeding")
        sys.exit(1)
    else:
        print("\n[FAIL] TEST 1 FAILED - Do not proceed without fixing BESS placement")
        sys.exit(2)
