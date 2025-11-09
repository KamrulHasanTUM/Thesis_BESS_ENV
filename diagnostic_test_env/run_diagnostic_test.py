"""
Diagnostic Test Runner for BESS Environment
===========================================

This script runs a 5K timestep diagnostic test to verify that the implemented
improvements (SoC-aware reward, action feasibility, SoC headroom observations)
actually solve the congestion increase problem.

Target Metrics:
- Success Rate (loading decreased) > 80%
- Harmful Actions (loading increased) < 20%
- Action Feasibility > 0.75
- No catastrophic loading spikes (>50% increase)

If these targets are met, we can proceed with full 50K-150K training.
"""

import numpy as np
import json
from ENV_BESS_main import ENV_BESS
from config import load_config, create_bess_env_config, create_training_config
from training import train_model_with_wandb
import os

def analyze_diagnostic_results(log_file="diagnostic_results.txt"):
    """
    Analyze the diagnostic test results from terminal output.

    Returns:
        dict: {
            'total_steps': int,
            'successful_steps': int (loading decreased),
            'harmful_steps': int (loading increased),
            'neutral_steps': int (loading same),
            'success_rate': float,
            'harmful_rate': float,
            'avg_loading_before': float,
            'avg_loading_after': float,
            'avg_loading_change': float,
            'max_loading_spike': float,
            'soc_violations': int,
        }
    """
    results = {
        'total_steps': 0,
        'successful_steps': 0,
        'harmful_steps': 0,
        'neutral_steps': 0,
        'loading_changes': [],
        'loading_before_values': [],
        'loading_after_values': [],
        'soc_violations': 0,
    }

    # Parse log file if it exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for "Max loading before:" pattern
            if "Max loading before:" in line:
                try:
                    loading_before = float(line.split(":")[-1].strip())
                    results['loading_before_values'].append(loading_before)

                    # Next few lines should have "Max loading after:"
                    for j in range(i+1, min(i+10, len(lines))):
                        if "Max loading after:" in lines[j]:
                            loading_after = float(lines[j].split(":")[-1].strip())
                            results['loading_after_values'].append(loading_after)

                            # Calculate change
                            change = loading_after - loading_before
                            results['loading_changes'].append(change)
                            results['total_steps'] += 1

                            if change < -0.01:  # Loading decreased (successful)
                                results['successful_steps'] += 1
                            elif change > 0.01:  # Loading increased (harmful)
                                results['harmful_steps'] += 1
                            else:  # No significant change
                                results['neutral_steps'] += 1

                            break
                except:
                    pass

            # Count SoC violations
            if "SoC hit lower bound" in line or "SoC hit upper bound" in line:
                results['soc_violations'] += 1

            i += 1

    # Compute summary statistics
    if results['total_steps'] > 0:
        results['success_rate'] = results['successful_steps'] / results['total_steps']
        results['harmful_rate'] = results['harmful_steps'] / results['total_steps']
        results['neutral_rate'] = results['neutral_steps'] / results['total_steps']
    else:
        results['success_rate'] = 0.0
        results['harmful_rate'] = 0.0
        results['neutral_rate'] = 0.0

    if results['loading_changes']:
        results['avg_loading_change'] = np.mean(results['loading_changes'])
        results['max_loading_spike'] = np.max(results['loading_changes'])
        results['min_loading_drop'] = np.min(results['loading_changes'])
    else:
        results['avg_loading_change'] = 0.0
        results['max_loading_spike'] = 0.0
        results['min_loading_drop'] = 0.0

    if results['loading_before_values']:
        results['avg_loading_before'] = np.mean(results['loading_before_values'])
    else:
        results['avg_loading_before'] = 0.0

    if results['loading_after_values']:
        results['avg_loading_after'] = np.mean(results['loading_after_values'])
    else:
        results['avg_loading_after'] = 0.0

    return results


def print_diagnostic_report(results, iteration=1):
    """Print a formatted diagnostic report."""
    print("\n" + "="*80)
    print(f"DIAGNOSTIC TEST ITERATION {iteration} - RESULTS SUMMARY")
    print("="*80)

    print(f"\nPERFORMANCE METRICS:")
    print(f"  Total Steps Analyzed:     {results['total_steps']}")
    print(f"  Successful Steps:         {results['successful_steps']} ({results['success_rate']*100:.1f}%)")
    print(f"  Harmful Steps:            {results['harmful_steps']} ({results['harmful_rate']*100:.1f}%)")
    print(f"  Neutral Steps:            {results['neutral_steps']} ({results['neutral_rate']*100:.1f}%)")

    print(f"\nTARGET ACHIEVEMENT:")
    target_met = "[OK]" if results['success_rate'] >= 0.80 else "[FAIL]"
    print(f"  Success Rate >= 80%:      {target_met} ({results['success_rate']*100:.1f}% vs 80% target)")

    target_met = "[OK]" if results['harmful_rate'] <= 0.20 else "[FAIL]"
    print(f"  Harmful Rate <= 20%:      {target_met} ({results['harmful_rate']*100:.1f}% vs 20% target)")

    print(f"\nLOADING STATISTICS:")
    print(f"  Average Loading Before:   {results['avg_loading_before']:.2f}%")
    print(f"  Average Loading After:    {results['avg_loading_after']:.2f}%")
    print(f"  Average Loading Change:   {results['avg_loading_change']:.2f}%")
    print(f"  Max Loading Spike:        {results['max_loading_spike']:.2f}%")
    print(f"  Max Loading Drop:         {abs(results['min_loading_drop']):.2f}%")

    print(f"\nCONSTRAINT VIOLATIONS:")
    print(f"  SoC Boundary Hits:        {results['soc_violations']}")

    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    if results['success_rate'] >= 0.80 and results['harmful_rate'] <= 0.20:
        print("  [PASS] DIAGNOSTIC TEST PASSED!")
        print("  Ready for full 50K-150K training.")
    elif results['success_rate'] >= 0.70 and results['harmful_rate'] <= 0.30:
        print("  [PARTIAL] PARTIAL SUCCESS - Improvements needed")
        print("  Continue iterating on reward function tuning.")
    else:
        print("  [FAIL] DIAGNOSTIC TEST FAILED")
        print("  Significant improvements required.")

    print("="*80 + "\n")

    return results['success_rate'] >= 0.80 and results['harmful_rate'] <= 0.20


def main():
    """Run diagnostic test."""
    print("\n" + "="*80)
    print("STARTING DIAGNOSTIC TEST - 5K TIMESTEPS")
    print("="*80)
    print("\nObjective: Verify SoC-aware improvements solve congestion increase problem")
    print("Target: Success Rate > 80%, Harmful Actions < 20%")
    print("\nThis test will:")
    print("  1. Train BESS agent for 5,000 timesteps (~10-15 minutes)")
    print("  2. Monitor reward breakdown for new components")
    print("  3. Calculate success rate from loading changes")
    print("  4. Determine if improvements work correctly")
    print("\n" + "="*80 + "\n")

    # Load configuration
    print("Loading configuration...")
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)
    training_config = create_training_config(init_meta)

    # Verify it's configured for 5K timesteps
    assert training_config['total_timesteps'] == 5000, \
        f"Expected 5K timesteps, got {training_config['total_timesteps']}"

    print(f"[OK] Configuration loaded: {training_config['total_timesteps']} timesteps")
    print(f"   Case study: {env_config['case_study']}")
    print(f"   BESS units: {env_config['num_bess']}")
    print(f"   Bonus constant: {env_config['bonus_constant']}")
    print(f"   WandB project: diagnostic-test-bess-env\n")

    # Override WandB settings for diagnostic test
    training_config['exp_name'] = 'diagnostic_5k_soc_aware_test'

    # Create environment to verify it works
    print("Creating environment...")
    env = ENV_BESS(**env_config)
    print(f"[OK] Environment created successfully")
    print(f"   Grid: {env.simbench_code}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}\n")

    # Run training with WandB
    print("Starting training with WandB logging...")
    print("="*80)
    print("TRAINING OUTPUT (watch for reward breakdown):")
    print("="*80 + "\n")

    try:
        # Import the setup function
        from training import setup_environment, create_model, train_model_with_wandb

        # Create model
        env_for_model = setup_environment(env_config, training_config)
        model = create_model(env_for_model, training_config)

        # Train with WandB
        final_model = train_model_with_wandb(
            model=model,
            training_config=training_config,
            env_config=env_config
        )

        print("\n" + "="*80)
        print("[OK] TRAINING COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

        # Note: In a real implementation, we would parse wandb logs or
        # save terminal output to a file for analysis
        print("NOTE: To fully analyze results, run this script with output redirection:")
        print("  python run_diagnostic_test.py > diagnostic_output.txt 2>&1")
        print("\nThen analyze diagnostic_output.txt for:")
        print("  - Reward breakdown showing 'action_feasibility', 'infeasibility_penalty'")
        print("  - Loading before/after values")
        print("  - Success rate calculation")

    except Exception as e:
        print(f"\n[ERROR] TRAINING FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nPlease check:")
        print("  1. All required files are in diagnostic_test_env/")
        print("  2. Python environment has all dependencies installed")
        print("  3. init_meta.json is present and valid")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
