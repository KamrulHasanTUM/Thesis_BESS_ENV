"""
Simple Diagnostic Test - Direct Environment Testing
Runs the environment directly to verify improvements are working
"""

import numpy as np
from ENV_BESS_main import ENV_BESS
from config import load_config, create_bess_env_config

def run_simple_diagnostic(num_episodes=100):
    """
    Run environment for N episodes and track success rate.

    Success = action reduces loading
    Harmful = action increases loading
    """
    # Load config
    print("="*80)
    print("SIMPLE DIAGNOSTIC TEST - DIRECT ENVIRONMENT VERIFICATION")
    print("="*80)
    print("\nObjective: Verify SoC-aware improvements are active")
    print(f"Episodes to run: {num_episodes}")
    print(f"Steps per episode: 50")
    print(f"Total steps: {num_episodes * 50}")
    print("\n" + "="*80 + "\n")

    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)

    # Create environment
    env = ENV_BESS(**env_config)
    print(f"[OK] Environment created")
    print(f"   Grid: {env.simbench_code}")
    print(f"   BESS units: {env.num_bess} at buses {env.bess_locations}")
    print(f"   Bonus constant: {env.bonus_constant}")

    # Check for improvements in observation space
    obs, _ = env.reset()
    has_headroom = 'bess_charge_headroom' in obs and 'bess_discharge_headroom' in obs
    has_soc_norm = 'bess_soc_normalized' in obs

    print(f"\n[VERIFICATION] Observation space includes:")
    print(f"   bess_charge_headroom: {'YES' if 'bess_charge_headroom' in obs else 'NO'}")
    print(f"   bess_discharge_headroom: {'YES' if 'bess_discharge_headroom' in obs else 'NO'}")
    print(f"   bess_soc_normalized: {'YES' if 'bess_soc_normalized' in obs else 'NO'}")

    if not (has_headroom and has_soc_norm):
        print("\n[ERROR] Missing expected observations! Improvements may not be active.")
        return None

    print("\n[OK] All expected observations present!\n")
    print("="*80)
    print("RUNNING EPISODES...")
    print("="*80 + "\n")

    # Track metrics
    results = {
        'total_steps': 0,
        'successful_steps': 0,  # Loading decreased
        'harmful_steps': 0,  # Loading increased
        'neutral_steps': 0,  # No change
        'loading_changes': [],
        'soc_violations': 0,
        'reward_has_feasibility': False,
        'reward_has_infeasibility_penalty': False,
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < 50:
            # Random action for diagnostic (agent not trained yet)
            action = env.action_space.sample()

            # Get loading before
            loading_before = env.net.res_line['loading_percent'].max()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get loading after
            loading_after = env.net.res_line['loading_percent'].max()

            # Track change
            change = loading_after - loading_before
            results['loading_changes'].append(change)
            results['total_steps'] += 1

            if change < -0.01:
                results['successful_steps'] += 1
            elif change > 0.01:
                results['harmful_steps'] += 1
            else:
                results['neutral_steps'] += 1

            #  Check reward breakdown for new components
            if hasattr(env, 'last_reward_breakdown'):
                if 'action_feasibility' in env.last_reward_breakdown:
                    results['reward_has_feasibility'] = True
                if 'infeasibility_penalty' in env.last_reward_breakdown:
                    results['reward_has_infeasibility_penalty'] = True

            step += 1

        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes...")

    # Calculate success rate
    if results['total_steps'] > 0:
        results['success_rate'] = results['successful_steps'] / results['total_steps']
        results['harmful_rate'] = results['harmful_steps'] / results['total_steps']
        results['neutral_rate'] = results['neutral_steps'] / results['total_steps']

    if results['loading_changes']:
        results['avg_loading_change'] = np.mean(results['loading_changes'])
        results['max_loading_spike'] = np.max(results['loading_changes'])
        results['min_loading_drop'] = np.min(results['loading_changes'])

    return results


def print_results(results):
    """Print formatted results."""
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST RESULTS")
    print("="*80)

    print(f"\nPERFORMANCE METRICS:")
    print(f"  Total Steps Analyzed:     {results['total_steps']}")
    print(f"  Successful Steps:         {results['successful_steps']} ({results['success_rate']*100:.1f}%)")
    print(f"  Harmful Steps:            {results['harmful_steps']} ({results['harmful_rate']*100:.1f}%)")
    print(f"  Neutral Steps:            {results['neutral_steps']} ({results['neutral_rate']*100:.1f}%)")

    print(f"\nTARGET ACHIEVEMENT (Random Policy Baseline):")
    target_met = "[OK]" if results['success_rate'] >= 0.45 else "[FAIL]"
    print(f"  Success Rate >= 45%:      {target_met} ({results['success_rate']*100:.1f}% vs 45% baseline)")
    print(f"  (Note: Random actions, so ~50% expected)")

    print(f"\nLOADING STATISTICS:")
    print(f"  Average Loading Change:   {results['avg_loading_change']:.2f}%")
    print(f"  Max Loading Spike:        {results['max_loading_spike']:.2f}%")
    print(f"  Max Loading Drop:         {abs(results['min_loading_drop']):.2f}%")

    print(f"\nREWARD FUNCTION VERIFICATION:")
    print(f"  Has action_feasibility:          {'YES' if results['reward_has_feasibility'] else 'NO'}")
    print(f"  Has infeasibility_penalty:       {'YES' if results['reward_has_infeasibility_penalty'] else 'NO'}")

    print("\nOVERALL ASSESSMENT:")
    if results['reward_has_feasibility'] and results['reward_has_infeasibility_penalty']:
        print("  [PASS] Improvements are ACTIVE in the code!")
        print("  Ready to proceed with full 5K timestep training.")
    else:
        print("  [FAIL] Improvements NOT detected in reward function.")
        print("  Check env_helpers.py calculate_bess_reward() function.")

    print("="*80 + "\n")


if __name__ == "__main__":
    results = run_simple_diagnostic(num_episodes=100)
    if results:
        print_results(results)
    else:
        print("\n[ERROR] Diagnostic test failed to run.")
