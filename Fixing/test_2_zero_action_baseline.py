"""
Test 2: Zero-Action Baseline (No BESS Intervention)

This test evaluates grid behavior when BESS does nothing (all units idle).
Helps establish:
1. How often does grid experience congestion without BESS?
2. What is the typical loading level?
3. Is BESS intervention actually needed?
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


def run_zero_action_baseline(num_episodes=100, verbose=True):
    """
    Test grid behavior with BESS doing nothing (all actions = 0).

    Args:
        num_episodes: Number of episodes to test
        verbose: Whether to print progress

    Returns:
        dict: Test results with statistics
    """
    print("="*80)
    print("TEST 2: ZERO-ACTION BASELINE (NO BESS INTERVENTION)")
    print("="*80)
    print(f"Testing grid behavior with BESS idle...")
    print(f"Episodes: {num_episodes}")
    print("-"*80)

    # Create environment
    env_config = get_test_env_config()
    env = ENV_BESS(**env_config)

    # Storage for results
    results = {
        'max_loadings': [],
        'avg_loadings': [],
        'congestion_episodes': 0,  # Episodes with loading > 80%
        'severe_congestion_episodes': 0,  # Episodes with loading > 100%
        'episode_stats': []
    }

    np.random.seed(TEST_PARAMS['random_seed'])

    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_loadings = []

        for step in range(env.max_step):
            # Zero action (all BESS idle)
            zero_action = np.zeros(env.num_bess)

            # Take step
            obs, reward, terminated, truncated, info = env.step(zero_action)

            # Get loading metrics
            try:
                max_loading = env.net.res_line['loading_percent'].max()
                avg_loading = env.net.res_line['loading_percent'].mean()

                episode_loadings.append(max_loading)
                results['max_loadings'].append(max_loading)
                results['avg_loadings'].append(avg_loading)

            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract loading at episode {episode}, step {step}: {e}")

            if terminated or truncated:
                break

        # Episode statistics
        if len(episode_loadings) > 0:
            max_loading_in_episode = np.max(episode_loadings)
            mean_loading_in_episode = np.mean(episode_loadings)

            episode_stat = {
                'episode': episode,
                'max_loading': float(max_loading_in_episode),
                'mean_loading': float(mean_loading_in_episode),
                'had_congestion': bool(max_loading_in_episode > 80),
                'had_severe_congestion': bool(max_loading_in_episode > 100)
            }

            results['episode_stats'].append(episode_stat)

            if max_loading_in_episode > 80:
                results['congestion_episodes'] += 1
            if max_loading_in_episode > 100:
                results['severe_congestion_episodes'] += 1

            if verbose and episode % 10 == 0:
                print(f"Episode {episode:3d}: Max Loading={max_loading_in_episode:.1f}%, "
                      f"Mean Loading={mean_loading_in_episode:.1f}%")

    # Calculate overall statistics
    overall_stats = {
        'mean_max_loading': float(np.mean(results['max_loadings'])),
        'std_max_loading': float(np.std(results['max_loadings'])),
        'median_max_loading': float(np.median(results['max_loadings'])),
        'percentile_95_loading': float(np.percentile(results['max_loadings'], 95)),
        'max_loading_observed': float(np.max(results['max_loadings'])),
        'mean_avg_loading': float(np.mean(results['avg_loadings'])),
        'congestion_rate': results['congestion_episodes'] / num_episodes,
        'severe_congestion_rate': results['severe_congestion_episodes'] / num_episodes,
        'num_episodes': num_episodes,
        'total_steps': len(results['max_loadings'])
    }

    # Print results
    print("\n" + "="*80)
    print("ZERO-ACTION BASELINE RESULTS")
    print("="*80)
    print(f"Mean Max Loading:         {overall_stats['mean_max_loading']:.2f}%")
    print(f"Std Dev Max Loading:      {overall_stats['std_max_loading']:.2f}%")
    print(f"Median Max Loading:       {overall_stats['median_max_loading']:.2f}%")
    print(f"95th Percentile Loading:  {overall_stats['percentile_95_loading']:.2f}%")
    print(f"Maximum Loading Observed: {overall_stats['max_loading_observed']:.2f}%")
    print(f"Mean Average Loading:     {overall_stats['mean_avg_loading']:.2f}%")
    print(f"Congestion Rate (>80%):   {overall_stats['congestion_rate']:.1%}")
    print(f"Severe Congestion (>100%):{overall_stats['severe_congestion_rate']:.1%}")
    print("-"*80)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-"*80)

    mean_loading = overall_stats['mean_max_loading']
    congestion_rate = overall_stats['congestion_rate']
    severe_rate = overall_stats['severe_congestion_rate']

    if mean_loading > 80 or congestion_rate > 0.5:
        print("[PASS] GOOD: Grid experiences significant congestion without BESS")
        print("   Conclusion: BESS intervention is needed and task has learning signal.")
        print("   Next steps: Proceed with improved training.")
        result_status = "PASS"

    elif 50 < mean_loading <= 80 and 0.2 <= congestion_rate <= 0.5:
        print("[WARNING]  MODERATE: Grid has occasional congestion")
        print("   Conclusion: BESS can help but task may be easier than expected.")
        print("   Next steps: Proceed but may need to adjust reward scaling.")
        result_status = "MODERATE"

    elif mean_loading <= 50 and congestion_rate < 0.2:
        print("[WARNING]  WARNING: Grid operates well without BESS")
        print("   Conclusion: Limited learning signal - task may be too easy.")
        print("   Next steps: Consider using high-loading scenarios only.")
        result_status = "EASY"

    else:
        print("[UNCLEAR] UNCLEAR: Results need manual review")
        result_status = "UNCLEAR"

    overall_stats['result_status'] = result_status

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(TEST_PARAMS['results_dir'], f'test_2_zero_action_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(overall_stats, f, indent=4)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    return overall_stats


if __name__ == "__main__":
    results = run_zero_action_baseline(num_episodes=TEST_PARAMS['num_episodes'])

    print(f"\n[PASS] TEST 2 COMPLETED - Status: {results['result_status']}")
    sys.exit(0)
