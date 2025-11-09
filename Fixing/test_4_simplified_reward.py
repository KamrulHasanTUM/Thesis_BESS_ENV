"""
Test 4: Simplified Reward Function

This test verifies that the simplified reward function (only congestion reduction)
provides clearer learning signal compared to the complex multi-component reward.
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
from improved_env_helpers import calculate_simplified_reward, calculate_improved_reward


def test_reward_functions(num_episodes=50, verbose=True):
    """
    Compare simplified vs improved reward functions.

    Args:
        num_episodes: Number of episodes to test
        verbose: Whether to print progress

    Returns:
        dict: Comparison results
    """
    print("="*80)
    print("TEST 4: SIMPLIFIED REWARD FUNCTION")
    print("="*80)
    print(f"Comparing reward functions over {num_episodes} episodes...")
    print("-"*80)

    # Create environment
    env_config = get_test_env_config()
    env = ENV_BESS(**env_config)

    # Storage for results
    results = {
        'simplified_rewards': [],
        'improved_rewards': [],
        'loading_deltas': [],
        'reward_examples': []
    }

    np.random.seed(TEST_PARAMS['random_seed'])

    print("\nTesting reward functions with random policy...")
    print("-"*80)

    for episode in range(num_episodes):
        obs, info = env.reset()

        for step in range(env.max_step):
            # Random action
            action = np.random.uniform(-1.0, 1.0, size=env.num_bess)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)

            # Get loading metrics
            try:
                loading_before = env.loading_before_action
                loading_after = env.loading_after_action

                # Calculate both reward types
                simplified_reward, simplified_breakdown = calculate_simplified_reward(
                    env, loading_before, loading_after
                )

                improved_reward, improved_breakdown = calculate_improved_reward(
                    env, loading_before, loading_after
                )

                # Store results
                results['simplified_rewards'].append(simplified_reward)
                results['improved_rewards'].append(improved_reward)
                results['loading_deltas'].append(loading_before - loading_after)

                # Store interesting examples
                if len(results['reward_examples']) < 20:
                    example = {
                        'episode': episode,
                        'step': step,
                        'loading_before': float(loading_before),
                        'loading_after': float(loading_after),
                        'delta': float(loading_before - loading_after),
                        'simplified_reward': float(simplified_reward),
                        'improved_reward': float(improved_reward),
                        'simplified_breakdown': {k: float(v) if isinstance(v, np.floating) else v
                                                for k, v in simplified_breakdown.items()},
                        'improved_breakdown': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                              for k, v in improved_breakdown.items()}
                    }
                    results['reward_examples'].append(example)

            except (AttributeError, Exception) as e:
                if verbose and step == 0:
                    print(f"Warning at episode {episode}: {e}")

            if terminated or truncated:
                break

        if verbose and episode % 10 == 0:
            print(f"Episode {episode} completed")

    # Calculate statistics
    stats = {
        'simplified': {
            'mean': float(np.mean(results['simplified_rewards'])),
            'std': float(np.std(results['simplified_rewards'])),
            'min': float(np.min(results['simplified_rewards'])),
            'max': float(np.max(results['simplified_rewards'])),
            'median': float(np.median(results['simplified_rewards']))
        },
        'improved': {
            'mean': float(np.mean(results['improved_rewards'])),
            'std': float(np.std(results['improved_rewards'])),
            'min': float(np.min(results['improved_rewards'])),
            'max': float(np.max(results['improved_rewards'])),
            'median': float(np.median(results['improved_rewards']))
        },
        'loading_delta': {
            'mean': float(np.mean(results['loading_deltas'])),
            'std': float(np.std(results['loading_deltas']))
        },
        'num_samples': len(results['simplified_rewards'])
    }

    # Calculate correlation with actual loading reduction
    simplified_corr = np.corrcoef(results['loading_deltas'], results['simplified_rewards'])[0, 1]
    improved_corr = np.corrcoef(results['loading_deltas'], results['improved_rewards'])[0, 1]

    stats['correlations'] = {
        'simplified_vs_delta': float(simplified_corr),
        'improved_vs_delta': float(improved_corr)
    }

    # Print results
    print("\n" + "="*80)
    print("REWARD FUNCTION COMPARISON")
    print("="*80)

    print("\nSIMPLIFIED REWARD (Only Congestion):")
    print(f"  Mean:       {stats['simplified']['mean']:+.2f}")
    print(f"  Std Dev:    {stats['simplified']['std']:.2f}")
    print(f"  Range:      [{stats['simplified']['min']:+.2f}, {stats['simplified']['max']:+.2f}]")
    print(f"  Correlation with Delta: {simplified_corr:.3f}")

    print("\nIMPROVED REWARD (Congestion + SoC Penalty):")
    print(f"  Mean:       {stats['improved']['mean']:+.2f}")
    print(f"  Std Dev:    {stats['improved']['std']:.2f}")
    print(f"  Range:      [{stats['improved']['min']:+.2f}, {stats['improved']['max']:+.2f}]")
    print(f"  Correlation with Delta: {improved_corr:.3f}")

    print(f"\nLoading Delta Statistics:")
    print(f"  Mean:       {stats['loading_delta']['mean']:+.3f}%")
    print(f"  Std Dev:    {stats['loading_delta']['std']:.3f}%")

    # Show examples
    print("\n" + "-"*80)
    print("EXAMPLE REWARD CALCULATIONS:")
    print("-"*80)
    for i, ex in enumerate(results['reward_examples'][:3], 1):
        print(f"\nExample {i}:")
        print(f"  Loading: {ex['loading_before']:.1f}% -> {ex['loading_after']:.1f}% (Delta={ex['delta']:+.2f}%)")
        print(f"  Simplified:  {ex['simplified_reward']:+.1f}")
        print(f"  Improved:    {ex['improved_reward']:+.1f} "
              f"(congestion: {ex['improved_breakdown']['congestion']:+.1f}, "
              f"soc_penalty: {ex['improved_breakdown']['soc_penalty']:+.1f})")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("-"*80)

    if abs(simplified_corr) > 0.9:
        print(f"[PASS] EXCELLENT: Simplified reward highly correlated with loading reduction ({simplified_corr:.3f})")
        print("   Conclusion: Simplified reward provides clean learning signal.")
        result_status = "EXCELLENT"
    elif abs(simplified_corr) > 0.7:
        print(f"[PASS] GOOD: Simplified reward well correlated with loading reduction ({simplified_corr:.3f})")
        result_status = "GOOD"
    else:
        print(f"[WARNING]  WARNING: Simplified reward weakly correlated ({simplified_corr:.3f})")
        print("   Conclusion: May need further reward adjustments.")
        result_status = "NEEDS_REVIEW"

    if abs(improved_corr) > abs(simplified_corr):
        print(f"\n[PASS] Improved reward has better correlation ({improved_corr:.3f} vs {simplified_corr:.3f})")
        print("   Recommendation: Use improved reward function.")
        recommended = "IMPROVED"
    else:
        print(f"\n[PASS] Simplified reward has better correlation ({simplified_corr:.3f} vs {improved_corr:.3f})")
        print("   Recommendation: Use simplified reward function.")
        recommended = "SIMPLIFIED"

    stats['result_status'] = result_status
    stats['recommended'] = recommended

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(TEST_PARAMS['results_dir'], f'test_4_reward_comparison_{timestamp}.json')

    save_data = {
        'statistics': stats,
        'examples': results['reward_examples'][:10]
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=4)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    return stats


if __name__ == "__main__":
    results = test_reward_functions(num_episodes=50)

    print(f"\n[PASS] TEST 4 COMPLETED")
    print(f"   Status: {results['result_status']}")
    print(f"   Recommended: {results['recommended']} reward function")
    sys.exit(0)
