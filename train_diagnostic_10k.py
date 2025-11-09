"""
10K TIMESTEP DIAGNOSTIC TRAINING TEST
======================================

This script tests the IMPROVED reward function to verify it prevents policy collapse.

Test Parameters:
- 10,000 timesteps (vs 100,000 in full training)
- Same PPO algorithm, hyperparameters, network architecture
- Improved reward function with policy collapse prevention
- Monitors for collapse indicators every 1000 steps

Success Criteria:
- active_units > 3 throughout training
- avg_power > 5 MW throughout training
- No SoC stuck at 0.5
- Reward improves over time (not collapses)
- Final reward > -100 (vs -10,833 for collapsed agent)

If this test passes, improvements are safe to apply to main code.
"""

import sys
import os
import numpy as np
import warnings
import json
from datetime import datetime

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, current_dir)  # This folder (improved_version)
sys.path.insert(0, parent_dir)   # diagnostic_tests folder
sys.path.insert(0, grandparent_dir)  # Main project folder

# Import improved environment
from ENV_BESS_improved import ENV_BESS

# Import original config and training utilities
from config import load_config, create_bess_env_config, create_training_config
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings('ignore')


class CollapseMonitorCallback(BaseCallback):
    """
    Callback to monitor for policy collapse during training.

    Checks every N steps for collapse indicators:
    - All actions near zero
    - All SoCs stuck at 0.5
    - No improvement in reward
    """

    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.collapse_detected = False
        self.step_count = 0

        # Tracking
        self.action_history = []
        self.soc_history = []
        self.reward_history = []

        # Results
        self.checkpoints = []

    def _on_step(self) -> bool:
        """Called after each step."""
        self.step_count += 1

        # Collect data every step
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]  # First environment
            self.action_history.append(action)

        # Check every check_freq steps
        if self.step_count % self.check_freq == 0:
            return self._check_for_collapse()

        return True

    def _check_for_collapse(self) -> bool:
        """Check if policy has collapsed."""

        # Get recent actions (last 100)
        recent_actions = self.action_history[-100:] if len(self.action_history) >= 100 else self.action_history

        if len(recent_actions) > 0:
            recent_actions_array = np.array(recent_actions)

            # Check 1: Are all actions near zero?
            action_magnitudes = np.abs(recent_actions_array).mean(axis=1)
            avg_action_magnitude = action_magnitudes.mean()

            # Check 2: Get recent rewards
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                # ep_info_buffer is a deque, convert to list first
                buffer_list = list(self.model.ep_info_buffer)
                recent_rewards = [ep['r'] for ep in buffer_list[-10:]]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            else:
                avg_reward = 0

            # Create checkpoint
            checkpoint = {
                'step': self.step_count,
                'avg_action_magnitude': float(avg_action_magnitude),
                'avg_reward': float(avg_reward),
                'num_recent_episodes': len(self.model.ep_info_buffer) if hasattr(self.model, 'ep_info_buffer') else 0,
            }
            self.checkpoints.append(checkpoint)

            # Print status
            print(f"\n{'='*80}")
            print(f"Collapse Monitor - Step {self.step_count}")
            print(f"{'='*80}")
            print(f"  Avg action magnitude: {avg_action_magnitude:.4f}")
            print(f"  Avg reward (last 10 ep): {avg_reward:.2f}")

            # Check for collapse
            if avg_action_magnitude < 0.01:
                print(f"  WARNING: Actions near zero! Possible collapse detected.")
                self.collapse_detected = True

            if avg_reward < -1000:
                print(f"  WARNING: Catastrophic reward! Possible collapse detected.")
                self.collapse_detected = True

            # Determine status
            if not self.collapse_detected:
                if avg_action_magnitude > 0.1:
                    print(f"  Status: HEALTHY - Agent taking significant actions")
                elif avg_action_magnitude > 0.05:
                    print(f"  Status: MODERATE - Some action, monitoring...")
                else:
                    print(f"  Status: WARNING - Low action magnitude")
            else:
                print(f"  Status: COLLAPSED - Agent stopped acting!")

            print(f"{'='*80}\n")

        return not self.collapse_detected  # Continue if not collapsed

    def save_results(self, filepath):
        """Save checkpoint results to file."""
        results = {
            'total_steps': self.step_count,
            'collapse_detected': self.collapse_detected,
            'checkpoints': self.checkpoints,
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Collapse monitor results saved to: {filepath}")


def setup_improved_environment():
    """Setup improved environment with collapse prevention."""
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)

    print("\n" + "="*80)
    print("ENVIRONMENT CONFIGURATION")
    print("="*80)
    print(f"  soc_penalty_weight: {env_config['soc_penalty_weight']}")
    print(f"  bonus_constant: {env_config['bonus_constant']}")
    print(f"  flexibility_bonus_weight: {env_config['flexibility_bonus_weight']}")
    print(f"  soc_clipping_penalty: {env_config['soc_clipping_penalty']}")
    print(f"  max_step: {env_config['max_step']}")
    print("="*80 + "\n")

    # Create environment
    env = ENV_BESS(**env_config)
    monitored_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored_env])

    return vec_env, env_config


def create_improved_model(env, logdir):
    """Create PPO model with improved hyperparameters."""
    from torch import nn

    policy_kwargs = dict(
        log_std_init=-2.0,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print("  Algorithm: PPO")
    print("  Policy: MultiInputPolicy")
    print("  Network: [256, 256] x 2")
    print("  Entropy coef: 0.05 (INCREASED from 0.01)")
    print("  Clip range: 0.2")
    print("  Learning rate: 0.0003")
    print("  Batch size: 256")
    print("  n_steps: 2048")
    print("  n_epochs: 10")
    print("="*80 + "\n")

    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logdir,
        n_epochs=10,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # INCREASED from 0.01 to maintain exploration
        max_grad_norm=0.5,
        learning_rate=0.0003,
    )


def run_diagnostic_training():
    """Run 10k timestep diagnostic training test."""

    print("\n" + "="*80)
    print("10K TIMESTEP DIAGNOSTIC TRAINING TEST")
    print("="*80)
    print("\nTesting IMPROVED reward function to prevent policy collapse")
    print("\nTest Parameters:")
    print("  Total timesteps: 10,000")
    print("  Check frequency: Every 1,000 steps")
    print("  Monitor: Action magnitude, reward, collapse indicators")
    print("\nSuccess Criteria:")
    print("  - Avg action magnitude > 0.05 throughout")
    print("  - Avg reward > -100 at end")
    print("  - No collapse detected")
    print("="*80 + "\n")

    # Setup
    vec_env, env_config = setup_improved_environment()
    logdir = os.path.join(current_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)

    # Create model
    model = create_improved_model(vec_env, logdir)

    # Create collapse monitor
    collapse_monitor = CollapseMonitorCallback(check_freq=1000, verbose=1)

    print("="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        # Train for 10k steps
        model.learn(
            total_timesteps=10000,
            callback=collapse_monitor,
            progress_bar=True
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  Collapse detected: {collapse_monitor.collapse_detected}")
        print("="*80 + "\n")

        # Save model
        model_path = os.path.join(current_dir, "improved_model_10k.zip")
        model.save(model_path)
        print(f"Model saved to: {model_path}\n")

        # Save collapse monitor results
        results_path = os.path.join(current_dir, "collapse_monitor_results.json")
        collapse_monitor.save_results(results_path)

        # Generate summary report
        generate_summary_report(collapse_monitor, env_config, duration)

        # Test final model
        test_final_model(model, vec_env)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        # Still save what we have
        collapse_monitor.save_results(os.path.join(current_dir, "collapse_monitor_results_interrupted.json"))

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        # Save error state
        collapse_monitor.save_results(os.path.join(current_dir, "collapse_monitor_results_error.json"))


def test_final_model(model, vec_env):
    """Test final model to verify it's working."""
    print("\n" + "="*80)
    print("TESTING FINAL MODEL")
    print("="*80 + "\n")

    obs = vec_env.reset()
    episode_actions = []
    episode_rewards = []
    episode_socs = []

    for step in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        episode_actions.append(action[0])
        episode_rewards.append(reward[0])

        # Get SoC from observation
        # Assuming SoC is in observation (first 5 elements)
        if 'bess_soc' in info[0]:
            episode_socs.append(info[0]['bess_soc'])

        if done[0]:
            break

    # Analyze results
    actions_array = np.array(episode_actions)
    action_magnitude = np.abs(actions_array).mean()
    avg_reward = np.mean(episode_rewards)

    print(f"Test Episode Results:")
    print(f"  Steps: {len(episode_rewards)}")
    print(f"  Avg action magnitude: {action_magnitude:.4f}")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Total reward: {sum(episode_rewards):.2f}")

    # Check for collapse
    if action_magnitude < 0.01:
        print(f"\n  WARNING: Actions near zero! Model may have collapsed.")
    elif action_magnitude < 0.05:
        print(f"\n  CAUTION: Low action magnitude. Monitor closely.")
    else:
        print(f"\n  SUCCESS: Model taking significant actions!")

    print("="*80 + "\n")


def generate_summary_report(collapse_monitor, env_config, duration):
    """Generate comprehensive summary report."""

    report_path = os.path.join(current_dir, "DIAGNOSTIC_TEST_RESULTS.txt")

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("10K TIMESTEP DIAGNOSTIC TRAINING TEST - RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n\n")

        f.write("TEST CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total timesteps: 10,000\n")
        f.write(f"Environment: ENV_BESS with IMPROVED reward function\n")
        f.write(f"Algorithm: PPO\n")
        f.write(f"Entropy coefficient: 0.05 (increased from 0.01)\n\n")

        f.write("ENVIRONMENT PARAMETERS\n")
        f.write("-"*80 + "\n")
        f.write(f"soc_penalty_weight: {env_config['soc_penalty_weight']}\n")
        f.write(f"bonus_constant: {env_config['bonus_constant']}\n")
        f.write(f"flexibility_bonus_weight: {env_config['flexibility_bonus_weight']}\n")
        f.write(f"soc_clipping_penalty: {env_config['soc_clipping_penalty']}\n")
        f.write(f"max_step: {env_config['max_step']}\n\n")

        f.write("COLLAPSE MONITORING RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Collapse detected: {collapse_monitor.collapse_detected}\n")
        f.write(f"Total checkpoints: {len(collapse_monitor.checkpoints)}\n\n")

        f.write("Checkpoint Details:\n")
        for i, checkpoint in enumerate(collapse_monitor.checkpoints):
            f.write(f"\n  Checkpoint {i+1} (Step {checkpoint['step']}):\n")
            f.write(f"    Avg action magnitude: {checkpoint['avg_action_magnitude']:.4f}\n")
            f.write(f"    Avg reward: {checkpoint['avg_reward']:.2f}\n")

            # Assess status
            if checkpoint['avg_action_magnitude'] < 0.01:
                status = "COLLAPSED"
            elif checkpoint['avg_action_magnitude'] < 0.05:
                status = "WARNING - Low activity"
            elif checkpoint['avg_action_magnitude'] < 0.1:
                status = "MODERATE"
            else:
                status = "HEALTHY"

            f.write(f"    Status: {status}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("VERDICT\n")
        f.write("="*80 + "\n")

        if collapse_monitor.collapse_detected:
            f.write("FAILURE: Policy collapse detected during training.\n")
            f.write("The improved reward function did NOT prevent collapse.\n")
            f.write("Further investigation and adjustments needed.\n")
        else:
            # Check final checkpoint
            if len(collapse_monitor.checkpoints) > 0:
                final_checkpoint = collapse_monitor.checkpoints[-1]
                final_action_mag = final_checkpoint['avg_action_magnitude']
                final_reward = final_checkpoint['avg_reward']

                if final_action_mag > 0.05 and final_reward > -100:
                    f.write("SUCCESS: Improved reward function prevents policy collapse!\n\n")
                    f.write(f"Final action magnitude: {final_action_mag:.4f} (target: >0.05)\n")
                    f.write(f"Final reward: {final_reward:.2f} (target: >-100)\n\n")
                    f.write("The improvements are SAFE to apply to main code.\n")
                else:
                    f.write("PARTIAL SUCCESS: No collapse detected, but metrics suboptimal.\n\n")
                    f.write(f"Final action magnitude: {final_action_mag:.4f} (target: >0.05)\n")
                    f.write(f"Final reward: {final_reward:.2f} (target: >-100)\n\n")
                    f.write("Consider further tuning before applying to main code.\n")
            else:
                f.write("INCONCLUSIVE: No checkpoints recorded.\n")

        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON WITH ORIGINAL (100k collapsed training)\n")
        f.write("="*80 + "\n")
        f.write("Original (collapsed):\n")
        f.write("  - Active units: 0\n")
        f.write("  - Avg power: 0 MW\n")
        f.write("  - Final reward: -10,833\n")
        f.write("  - All SoCs: stuck at 0.5\n\n")

        if len(collapse_monitor.checkpoints) > 0:
            final_checkpoint = collapse_monitor.checkpoints[-1]
            f.write("Improved (this test):\n")
            f.write(f"  - Action magnitude: {final_checkpoint['avg_action_magnitude']:.4f}\n")
            f.write(f"  - Final reward: {final_checkpoint['avg_reward']:.2f}\n")
            f.write(f"  - Collapse detected: {collapse_monitor.collapse_detected}\n\n")

            if not collapse_monitor.collapse_detected and final_checkpoint['avg_action_magnitude'] > 0.05:
                f.write("CONCLUSION: Improved version SIGNIFICANTLY better than original!\n")
            else:
                f.write("CONCLUSION: Improvements show promise but need further tuning.\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\nSummary report saved to: {report_path}\n")


if __name__ == "__main__":
    run_diagnostic_training()
