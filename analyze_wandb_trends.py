"""
analyze_wandb_trends.py

Analyzes WandB training trends to determine if 50K training was successful.
This script helps answer the critical questions:
1. Did BESS power trend downward, stay constant, or increase?
2. Did success rate improve above 50% baseline?
3. Was the final episode (0 MW) an anomaly or representative of convergence?
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_50k_training(run_path="535jye1k"):
    """
    Analyze the 50K training run to understand actual trends.

    Args:
        run_path: WandB run ID (default: 535jye1k for the 50K run)
    """
    print("=" * 80)
    print("50K TRAINING TREND ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing WandB run: {run_path}")
    print(f"Training duration: 51,200 timesteps")
    print()

    # Initialize WandB API
    api = wandb.Api()

    # Get the run (adjust project name if needed)
    try:
        # Try to get run from local wandb directory
        run = api.run(f"your-entity/your-project/{run_path}")
    except:
        print("⚠️  Could not access WandB API. Analyzing local files instead...")
        analyze_from_local_files(run_path)
        return

    # Get history data
    history = run.history()

    print("\n" + "=" * 80)
    print("SECTION 1: BESS POWER TRENDS")
    print("=" * 80)

    # Analyze BESS power over time
    if 'bess/avg_power' in history.columns:
        power_data = history['bess/avg_power'].dropna()

        # Divide into segments
        n = len(power_data)
        first_10k = power_data.iloc[:int(n*0.2)]
        middle = power_data.iloc[int(n*0.4):int(n*0.6)]
        last_10k = power_data.iloc[int(n*0.8):]

        print(f"\n📊 BESS Average Power Analysis:")
        print(f"   First 20% of training:  {first_10k.mean():.2f} MW (std: {first_10k.std():.2f})")
        print(f"   Middle 20% of training: {middle.mean():.2f} MW (std: {middle.std():.2f})")
        print(f"   Last 20% of training:   {last_10k.mean():.2f} MW (std: {last_10k.std():.2f})")
        print(f"   Overall average:        {power_data.mean():.2f} MW")
        print(f"   Overall range:          [{power_data.min():.2f}, {power_data.max():.2f}] MW")

        # Determine trend
        if last_10k.mean() < first_10k.mean() * 0.5:
            print(f"\n❌ CONCERNING: Power decreased by {(1 - last_10k.mean()/first_10k.mean())*100:.1f}%")
            print("   Agent is converging toward lower activity!")
        elif last_10k.mean() > first_10k.mean() * 1.2:
            print(f"\n✅ GOOD: Power increased by {(last_10k.mean()/first_10k.mean() - 1)*100:.1f}%")
            print("   Agent is becoming more active over time!")
        else:
            print(f"\n⚠️  NEUTRAL: Power relatively stable (±20%)")
            print("   Agent maintaining consistent activity level")

    print("\n" + "=" * 80)
    print("SECTION 2: SUCCESS RATE TRENDS")
    print("=" * 80)

    # Analyze success rate over time
    if 'congestion_episode/success_rate' in history.columns:
        success_data = history['congestion_episode/success_rate'].dropna()

        # Divide into segments
        n = len(success_data)
        first_10k = success_data.iloc[:int(n*0.2)]
        last_10k = success_data.iloc[int(n*0.8):]

        print(f"\n📊 Success Rate Analysis:")
        print(f"   First 20% of training: {first_10k.mean():.2f}% (std: {first_10k.std():.2f})")
        print(f"   Last 20% of training:  {last_10k.mean():.2f}% (std: {last_10k.std():.2f})")
        print(f"   Overall average:       {success_data.mean():.2f}%")
        print(f"   Final value:           {success_data.iloc[-1]:.2f}%")

        improvement = last_10k.mean() - first_10k.mean()

        if last_10k.mean() > 55 and improvement > 2:
            print(f"\n✅ EXCELLENT: Success rate improved by {improvement:.2f}%")
            print(f"   Above baseline (50%) and improving!")
        elif last_10k.mean() > 55:
            print(f"\n✅ GOOD: Success rate at {last_10k.mean():.2f}% (above 55% threshold)")
            print("   Agent is better than random baseline")
        elif abs(last_10k.mean() - 50) < 2:
            print(f"\n❌ CONCERNING: Success rate at {last_10k.mean():.2f}% (random baseline)")
            print("   Agent performing no better than random actions")
        else:
            print(f"\n⚠️  UNCLEAR: Success rate at {last_10k.mean():.2f}%")

    print("\n" + "=" * 80)
    print("SECTION 3: REWARD TRENDS")
    print("=" * 80)

    # Analyze reward over time
    if 'rollout/ep_reward_mean' in history.columns:
        reward_data = history['rollout/ep_reward_mean'].dropna()

        n = len(reward_data)
        first_10k = reward_data.iloc[:int(n*0.2)]
        last_10k = reward_data.iloc[int(n*0.8):]

        print(f"\n📊 Episode Reward Analysis:")
        print(f"   First 20% of training: {first_10k.mean():.2f} (std: {first_10k.std():.2f})")
        print(f"   Last 20% of training:  {last_10k.mean():.2f} (std: {last_10k.std():.2f})")
        print(f"   Final value:           {reward_data.iloc[-1]:.2f}")
        print(f"   Improvement:           {last_10k.mean() - first_10k.mean():.2f}")

        if last_10k.mean() > first_10k.mean():
            print(f"\n✅ Reward increased by {last_10k.mean() - first_10k.mean():.2f}")
            print("   But remember: High reward doesn't always mean good performance!")
            print("   Check if power stayed high while reward increased")

    print("\n" + "=" * 80)
    print("SECTION 4: FINAL EPISODE ANALYSIS")
    print("=" * 80)

    # Compare final episode to average
    final_power = history['bess/avg_power'].iloc[-1] if 'bess/avg_power' in history.columns else 0
    avg_power_last_100 = history['bess/avg_power'].iloc[-100:].mean() if 'bess/avg_power' in history.columns else 0

    print(f"\n📊 Final Episode vs Last 100 Episodes Average:")
    print(f"   Final episode power:     {final_power:.2f} MW")
    print(f"   Last 100 episodes avg:   {avg_power_last_100:.2f} MW")

    if abs(final_power) < 0.1 and avg_power_last_100 > 5:
        print(f"\n✅ GOOD NEWS: Final episode (0 MW) is an ANOMALY!")
        print(f"   Last 100 episodes averaged {avg_power_last_100:.2f} MW")
        print("   Agent is active on average - single idle episode is not concerning")
    elif abs(final_power) < 0.1 and avg_power_last_100 < 2:
        print(f"\n❌ CONCERNING: Final episode represents converged policy")
        print(f"   Last 100 episodes only averaged {avg_power_last_100:.2f} MW")
        print("   Agent has converged to near-zero activity")

    print("\n" + "=" * 80)
    print("SECTION 5: OVERALL VERDICT")
    print("=" * 80)

    # Make recommendation
    if 'bess/avg_power' in history.columns and 'congestion_episode/success_rate' in history.columns:
        power_data = history['bess/avg_power'].dropna()
        success_data = history['congestion_episode/success_rate'].dropna()

        avg_power = power_data.mean()
        avg_success = success_data.mean()

        print(f"\n📊 Overall Training Metrics:")
        print(f"   Average BESS Power:  {avg_power:.2f} MW")
        print(f"   Average Success Rate: {avg_success:.2f}%")

        if avg_power > 10 and avg_success > 55:
            print("\n✅ RECOMMENDATION: PROCEED TO 200K TRAINING")
            print("   Agent is active AND improving success rate")
            print("   Training is progressing in the right direction")
        elif avg_power > 10 and avg_success < 52:
            print("\n⚠️  RECOMMENDATION: INVESTIGATE REWARD FUNCTION")
            print("   Agent is active but not improving success rate")
            print("   May need reward tuning before extending training")
        elif avg_power < 5:
            print("\n❌ RECOMMENDATION: FIX REWARD FUNCTION BEFORE PROCEEDING")
            print("   Agent converging to low activity")
            print("   Extending training will likely make this worse")
        else:
            print("\n⚠️  RECOMMENDATION: UNCLEAR - Need more investigation")
            print("   Review the detailed metrics above")

    print("\n" + "=" * 80)
    print()

    # Create visualization
    create_trend_plots(history)


def analyze_from_local_files(run_id):
    """Analyze from local WandB files when API is not available."""
    print("\n📁 Analyzing from local files...")

    run_dir = Path(f"wandb/run-20251102_212152-{run_id}")

    if not run_dir.exists():
        print(f"❌ Could not find run directory: {run_dir}")
        print("\nPlease check WandB UI manually for these metrics:")
        print("1. Plot 'bess/avg_power' over all timesteps")
        print("2. Plot 'congestion_episode/success_rate' over all timesteps")
        print("3. Check if power is decreasing, stable, or increasing")
        return

    print(f"✅ Found run directory: {run_dir}")
    print("\n⚠️  Local file analysis is limited. For best results:")
    print("   1. Open WandB UI in browser")
    print("   2. Navigate to run: 535jye1k")
    print("   3. Check these plots:")
    print("      - bess/avg_power (should show trend over 51K steps)")
    print("      - congestion_episode/success_rate")
    print("      - rollout/ep_reward_mean")


def create_trend_plots(history):
    """Create visualization of key trends."""
    print("\n📊 Creating trend visualization plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('50K Training Trends Analysis', fontsize=16, fontweight='bold')

    # Plot 1: BESS Power over time
    if 'bess/avg_power' in history.columns:
        ax = axes[0, 0]
        power_data = history['bess/avg_power'].dropna()
        ax.plot(power_data.index, power_data.values, alpha=0.6, linewidth=0.5)
        ax.plot(power_data.rolling(100).mean(), 'r-', linewidth=2, label='Rolling avg (100)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('BESS Average Power (MW)')
        ax.set_title('BESS Power Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Success Rate over time
    if 'congestion_episode/success_rate' in history.columns:
        ax = axes[0, 1]
        success_data = history['congestion_episode/success_rate'].dropna()
        ax.plot(success_data.index, success_data.values, alpha=0.6, linewidth=0.5)
        ax.plot(success_data.rolling(50).mean(), 'b-', linewidth=2, label='Rolling avg (50)')
        ax.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Random baseline (50%)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Reward over time
    if 'rollout/ep_reward_mean' in history.columns:
        ax = axes[1, 0]
        reward_data = history['rollout/ep_reward_mean'].dropna()
        ax.plot(reward_data.index, reward_data.values, alpha=0.6, linewidth=0.5)
        ax.plot(reward_data.rolling(50).mean(), 'g-', linewidth=2, label='Rolling avg (50)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Reward Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Power vs Success Rate (correlation)
    if 'bess/avg_power' in history.columns and 'congestion_episode/success_rate' in history.columns:
        ax = axes[1, 1]
        # Align the data
        merged = pd.concat([
            history['bess/avg_power'],
            history['congestion_episode/success_rate']
        ], axis=1).dropna()

        if len(merged) > 0:
            ax.scatter(merged['bess/avg_power'], merged['congestion_episode/success_rate'],
                      alpha=0.3, s=10)
            ax.set_xlabel('BESS Average Power (MW)')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Power vs Success Rate Correlation')
            ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random baseline')
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = '50k_training_trends_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_file}")

    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WandB 50K Training Trend Analyzer")
    print("=" * 80)
    print("\nThis script analyzes the actual training trends to answer:")
    print("1. Was the final episode (0 MW) an anomaly or representative?")
    print("2. Did BESS power stay active or trend toward zero?")
    print("3. Did success rate improve above 50% baseline?")
    print("4. Should we proceed to 200K training?")
    print()

    analyze_50k_training()
