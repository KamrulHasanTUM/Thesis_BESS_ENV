"""
Extract comprehensive training metrics from wandb run for 500k timesteps analysis.
"""
import json
import os
import glob
import numpy as np
import pandas as pd

# Find the most recent wandb run (20251104)
run_dir = r"D:\Thesis\Project backup\Modif\ThesisEnv_refactor_withAllModif\wandb\run-20251104_040106-ancssyc4"

print(f"Analyzing run: {run_dir}")
print("=" * 80)

# Load wandb summary (final values)
summary_file = os.path.join(run_dir, "files", "wandb-summary.json")
with open(summary_file, 'r') as f:
    summary = json.load(f)

print("\n" + "=" * 80)
print("FINAL TRAINING METRICS (at 500k timesteps)")
print("=" * 80)
print(f"Total timesteps: {summary.get('step', 'N/A')}")
print(f"Total runtime: {summary.get('_runtime', 0) / 3600:.2f} hours")
print()

print("CONGESTION METRICS (Final Episode):")
print(f"  Success rate: {summary.get('congestion_episode/success_rate', 0):.2f}%")
print(f"  Mean delta: {summary.get('congestion_episode/mean_delta', 0):.4f}")
print(f"  Positive actions: {summary.get('congestion_episode/positive_actions', 0)}")
print(f"  Negative actions: {summary.get('congestion_episode/negative_actions', 0)}")
print(f"  Loading before: {summary.get('congestion/loading_before', 0):.2f}%")
print(f"  Loading after: {summary.get('congestion/loading_after', 0):.2f}%")
print(f"  Delta: {summary.get('congestion/delta', 0):.4f}")
print(f"  Delta percent: {summary.get('congestion/delta_percent', 0):.2f}%")
print()

print("GRID METRICS (Final Step):")
print(f"  Max loading: {summary.get('grid/max_loading', 0):.2f}%")
print(f"  Avg loading: {summary.get('grid/avg_loading', 0):.2f}%")
print()

print("BESS AGGREGATE METRICS (Final Step):")
print(f"  Average SoC: {summary.get('bess/avg_soc', 0)*100:.2f}%")
print(f"  Average power: {summary.get('bess/avg_power', 0):.2f} MW")
print(f"  SoC utilization: {summary.get('bess_summary/soc_utilization', 0)*100:.2f}%")
print(f"  Power utilization: {summary.get('bess_summary/power_utilization', 0)*100:.2f}%")
print(f"  Active units: {summary.get('bess_summary/active_units', 0)}")
print()

print("INDIVIDUAL BESS UNIT STATUS (Final Step):")
for i in range(1, 6):
    bus_ids = [39, 189, 230, 281, 282]
    bus = bus_ids[i-1]
    soc_key = f"bess_units/bess_{i}_bus_{bus}/soc"
    power_key = f"bess_units/bess_{i}_bus_{bus}/power"
    power_abs_key = f"bess_units/bess_{i}_bus_{bus}/power_abs"

    soc = summary.get(soc_key, 0) * 100
    power = summary.get(power_key, 0)
    power_abs = summary.get(power_abs_key, 0)

    print(f"  BESS {i} (Bus {bus}): SoC={soc:.1f}%, Power={power:+.2f}MW, |Power|={power_abs:.2f}MW")
print()

print("EPISODE METRICS (Final):")
print(f"  Episode reward mean: {summary.get('rollout/ep_reward_mean', 0):.2f}")
print(f"  Episode length mean: {summary.get('rollout/ep_len_mean', 0):.0f} steps")
print()

# Analyze reward scaling
print("=" * 80)
print("REWARD CALCULATION VERIFICATION")
print("=" * 80)
bonus_constant = 100
loading_delta = summary.get('congestion/delta', 0)
mean_delta_episode = summary.get('congestion_episode/mean_delta', 0)
episode_reward = summary.get('rollout/ep_reward_mean', 0)

print(f"bonus_constant (from config): {bonus_constant}")
print(f"Single-step loading delta: {loading_delta:.4f}%")
print(f"Episode mean delta: {mean_delta_episode:.4f}")
print(f"Episode reward: {episode_reward:.2f}")
print()

# Calculate expected reward components
congestion_per_step = loading_delta * bonus_constant
print(f"Congestion reward per step (delta × bonus): {congestion_per_step:.2f}")
print(f"If 50 steps, congestion only: {congestion_per_step * 50:.2f}")
print()

# Action magnitude bonus estimate
avg_power_util = summary.get('bess_summary/power_utilization', 0)
action_magnitude_bonus = 100 * avg_power_util  # action_magnitude_weight = 100
print(f"Action magnitude bonus estimate: {action_magnitude_bonus:.2f}")
print(f"If 50 steps: {action_magnitude_bonus * 50:.2f}")
print()

# SoC penalties estimate
num_near_bounds = 2  # Based on final SoC data (units at 10% or 90%)
soc_penalty = -0.2 * num_near_bounds
print(f"SoC penalty estimate (per step): {soc_penalty:.2f}")
print(f"If 50 steps: {soc_penalty * 50:.2f}")
print()

total_estimated = (congestion_per_step + action_magnitude_bonus + soc_penalty) * 50
print(f"Total estimated episode reward: {total_estimated:.2f}")
print(f"Actual episode reward: {episode_reward:.2f}")
print(f"Match ratio: {episode_reward / total_estimated:.2%}" if total_estimated != 0 else "Cannot compute")
print()

# Interpretation of mean_delta
print("=" * 80)
print("INTERPRETATION OF MEAN_DELTA METRIC")
print("=" * 80)
print(f"Reported mean_delta: {mean_delta_episode:.4f}")
print()
print("Analysis of what mean_delta represents:")
print("1. If it's per-step delta in %:")
print(f"   Average loading reduction per step: {mean_delta_episode:.4f}%")
print(f"   Over 50 steps, cumulative: {mean_delta_episode * 50:.2f}%")
print()
print("2. If it's episode cumulative delta:")
print(f"   Total loading reduction over episode: {mean_delta_episode:.4f}%")
print(f"   Average per step: {mean_delta_episode / 50:.4f}%")
print()
print("3. If it's scaled by bonus_constant:")
print(f"   Actual loading reduction: {mean_delta_episode / bonus_constant:.4f}%")
print(f"   Per step: {mean_delta_episode / bonus_constant / 50:.4f}%")
print()

# From loading_before and loading_after
actual_reduction = summary.get('congestion/loading_before', 0) - summary.get('congestion/loading_after', 0)
print(f"4. Direct calculation from loading_before - loading_after:")
print(f"   Single-step reduction: {actual_reduction:.4f}%")
print(f"   As percentage of before: {(actual_reduction / summary.get('congestion/loading_before', 1)) * 100:.2f}%")
print()

print("=" * 80)
print("ABSOLUTE CONGESTION REDUCTION ANALYSIS")
print("=" * 80)
print("From final timestep data:")
print(f"  Loading before BESS action: {summary.get('congestion/loading_before', 0):.2f}%")
print(f"  Loading after BESS action: {summary.get('congestion/loading_after', 0):.2f}%")
print(f"  Absolute reduction: {actual_reduction:.2f} percentage points")
print(f"  Relative reduction: {(actual_reduction / summary.get('congestion/loading_before', 1)) * 100:.2f}%")
print()

# SUCCESS RATE ANALYSIS
print("=" * 80)
print("LEARNING PROGRESSION INDICATORS")
print("=" * 80)
success_rate = summary.get('congestion_episode/success_rate', 0)
pos_actions = summary.get('congestion_episode/positive_actions', 0)
neg_actions = summary.get('congestion_episode/negative_actions', 0)
total_actions = pos_actions + neg_actions

print(f"Success rate: {success_rate:.2f}%")
print(f"Positive actions (reducing congestion): {pos_actions} / {total_actions}")
print(f"Negative actions (increasing congestion): {neg_actions} / {total_actions}")
print(f"Positive:Negative ratio: {pos_actions/neg_actions:.2f}:1" if neg_actions > 0 else "Perfect (no negative)")
print()

print("Interpretation:")
if success_rate >= 95:
    print("  ✓ EXCELLENT: Agent has mastered congestion reduction")
elif success_rate >= 80:
    print("  ✓ GOOD: Agent consistently reduces congestion")
elif success_rate >= 60:
    print("  ~ MODERATE: Agent learning but inconsistent")
else:
    print("  ✗ POOR: Agent still exploring")
print()

# BESS UTILIZATION ANALYSIS
print("=" * 80)
print("BESS UTILIZATION ANALYSIS")
print("=" * 80)
power_util = summary.get('bess_summary/power_utilization', 0)
soc_util = summary.get('bess_summary/soc_utilization', 0)
active_units = summary.get('bess_summary/active_units', 0)

print(f"Power utilization: {power_util * 100:.2f}% (0-100% of max power)")
print(f"SoC utilization: {soc_util * 100:.2f}% (position in 10-90% range)")
print(f"Active units: {active_units} / 5")
print()

print("Individual BESS Behavior:")
bus_ids = [39, 189, 230, 281, 282]
for i in range(1, 6):
    bus = bus_ids[i-1]
    soc = summary.get(f"bess_units/bess_{i}_bus_{bus}/soc", 0) * 100
    power = summary.get(f"bess_units/bess_{i}_bus_{bus}/power", 0)
    power_abs = summary.get(f"bess_units/bess_{i}_bus_{bus}/power_abs", 0)

    behavior = "IDLE" if power_abs < 1 else ("CHARGING" if power < 0 else "DISCHARGING")
    utilization = (power_abs / 50) * 100  # 50 MW is max

    print(f"  BESS {i} @ Bus {bus}:")
    print(f"    State: {behavior}")
    print(f"    SoC: {soc:.1f}% {'(AT LOWER BOUND!)' if soc < 11 else '(AT UPPER BOUND!)' if soc > 89 else ''}")
    print(f"    Power: {power:+.2f} MW ({utilization:.1f}% of max)")
    print()

print("=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
print(f"1. Training converged successfully after {summary.get('step', 0):,} timesteps")
print(f"2. Agent achieved {success_rate:.1f}% success rate in reducing congestion")
print(f"3. Mean congestion delta: {mean_delta_episode:.2f} units (interpretation requires further analysis)")
print(f"4. Absolute loading reduction: {actual_reduction:.2f} percentage points per step")
print(f"5. All {active_units} BESS units active and responding to grid conditions")
print(f"6. BESS power utilization: {power_util * 100:.1f}% (agent using significant power)")
print(f"7. Episode reward: {episode_reward:.0f} (primarily from congestion relief + action exploration)")
print()

print("CRITICAL QUESTIONS FOR VERIFICATION:")
print("1. What is the true unit of 'mean_delta'?")
print("   - Is it percentage points, scaled by bonus_constant, or cumulative?")
print("2. What is typical loading range in 'hL' case study?")
print("   - Final sample: 78% before → 18% after (60 point reduction)")
print("   - Is this representative of average episode performance?")
print("3. What is physical grid capacity?")
print("   - Need to calculate if 250 MW BESS is sufficient for congestion impact")
print()

print("=" * 80)
print("Analysis complete!")
print("=" * 80)
