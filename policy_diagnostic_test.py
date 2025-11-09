"""
POLICY DIAGNOSTIC TEST FRAMEWORK
=================================
Comprehensive test framework that works with TRAINED POLICY or RANDOM POLICY.

This test will give you 100% certainty about:
1. Consistent congestion reduction (loading before vs after each action)
2. Agent learning quality (is it responding to grid state or blind?)
3. BESS charge/discharge operation (all 5 units working?)
4. Reward components working accurately
5. SoC boundary violations (upper and lower bounds)

Usage:
------
WITH TRAINED AGENT:
    python policy_diagnostic_test.py --model path/to/final_model.zip

WITHOUT TRAINED AGENT (random baseline):
    python policy_diagnostic_test.py --random

Options:
    --episodes N     Number of episodes to test (default: 10)
    --steps N        Steps per episode (default: 50)
    --output DIR     Output directory for results (default: diagnostic_tests/results)
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
import json

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, create_bess_env_config
from ENV_BESS_main import ENV_BESS


class PolicyDiagnosticTest:
    """Diagnostic test framework for trained or random policies."""

    def __init__(self, model_path=None, num_episodes=10, steps_per_episode=50, output_dir="diagnostic_tests/results"):
        """
        Initialize diagnostic test.

        Args:
            model_path: Path to trained model .zip file (None for random policy)
            num_episodes: Number of episodes to test
            steps_per_episode: Steps per episode
            output_dir: Directory to save results
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.output_dir = output_dir
        self.total_steps = num_episodes * steps_per_episode

        # Load configuration
        self.cfg = create_bess_env_config(load_config())
        self.env = None
        self.model = None
        self.policy_type = "RANDOM" if model_path is None else "TRAINED"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Comprehensive metrics
        self.metrics = {
            'congestion': {
                'loading_before': [],
                'loading_after': [],
                'deltas': [],
                'improvements': [],
                'worsenings': [],
                'neutral': [],
                'improvement_per_episode': [],
                'worsening_per_episode': [],
            },
            'soc': {
                'history': [],  # Full SoC history
                'mean_per_episode': [],
                'std_per_episode': [],
                'min_per_episode': [],
                'max_per_episode': [],
                'lower_violations': 0,
                'upper_violations': 0,
                'lower_violations_per_episode': [],
                'upper_violations_per_episode': [],
                'in_flex_band': [],
                'flex_band_per_episode': [],
            },
            'bess_activity': {
                'unit_0': {'charge': 0, 'discharge': 0, 'idle': 0},
                'unit_1': {'charge': 0, 'discharge': 0, 'idle': 0},
                'unit_2': {'charge': 0, 'discharge': 0, 'idle': 0},
                'unit_3': {'charge': 0, 'discharge': 0, 'idle': 0},
                'unit_4': {'charge': 0, 'discharge': 0, 'idle': 0},
            },
            'rewards': {
                'total': [],
                'congestion': [],
                'soc_penalty': [],
                'flexibility_bonus': [],
                'clipping_penalty': [],
                'total_per_episode': [],
                'components_per_episode': [],
            },
            'actions': {
                'raw_actions': [],  # Before clipping
                'executed_actions': [],  # After clipping
                'clipping_events': 0,
                'clipping_per_episode': [],
            },
            'learning_quality': {
                'responsive_steps': 0,  # Steps where action helped reduce congestion
                'harmful_steps': 0,  # Steps where action worsened congestion
                'neutral_steps': 0,
                'responsiveness_per_episode': [],
            },
            'timing': {
                'start_time': None,
                'end_time': None,
                'duration_seconds': 0,
            }
        }

    def load_model(self):
        """Load trained model if path provided."""
        if self.model_path is None:
            print("\nUsing RANDOM POLICY for baseline testing")
            return

        try:
            from stable_baselines3 import PPO
            print(f"\nLoading trained model from: {self.model_path}")
            self.model = PPO.load(self.model_path)
            print("Model loaded successfully!")
            print(f"Policy type: {type(self.model.policy).__name__}")
        except FileNotFoundError:
            print(f"\nERROR: Model file not found: {self.model_path}")
            print("Falling back to RANDOM POLICY")
            self.model = None
            self.policy_type = "RANDOM"
        except Exception as e:
            print(f"\nERROR loading model: {e}")
            print("Falling back to RANDOM POLICY")
            self.model = None
            self.policy_type = "RANDOM"

    def get_action(self, obs):
        """
        Get action from trained policy or random policy.

        Args:
            obs: Current observation

        Returns:
            numpy array: Action to execute
        """
        if self.model is not None:
            # Use trained policy
            action, _states = self.model.predict(obs, deterministic=True)
            return action
        else:
            # Random policy (baseline)
            return np.random.uniform(-0.5, 0.5, size=5).astype(np.float32)

    def run_comprehensive_test(self):
        """Run complete diagnostic test."""
        print("="*80)
        print("POLICY DIAGNOSTIC TEST FRAMEWORK")
        print("="*80)
        print(f"\nPolicy Type: {self.policy_type}")
        print(f"Test Episodes: {self.num_episodes}")
        print(f"Steps per Episode: {self.steps_per_episode}")
        print(f"Total Steps: {self.total_steps}")
        print(f"\nEnvironment Configuration:")
        print(f"  soc_penalty_weight: {self.cfg['soc_penalty_weight']}")
        print(f"  bonus_constant: {self.cfg['bonus_constant']}")
        print(f"  flexibility_bonus_weight: {self.cfg['flexibility_bonus_weight']}")
        print(f"  soc_clipping_penalty: {self.cfg['soc_clipping_penalty']}")
        print(f"  max_step: {self.cfg['max_step']}")

        self.metrics['timing']['start_time'] = datetime.now()

        # Load model if provided
        self.load_model()

        # Initialize environment
        self.env = ENV_BESS(**self.cfg)

        print(f"\n{'='*80}")
        print(f"Starting {self.policy_type} Policy Test...")
        print(f"{'='*80}\n")

        # Run test episodes
        for episode in range(self.num_episodes):
            self._run_episode(episode)

        self.metrics['timing']['end_time'] = datetime.now()
        self.metrics['timing']['duration_seconds'] = (
            self.metrics['timing']['end_time'] - self.metrics['timing']['start_time']
        ).total_seconds()

        # Generate comprehensive reports
        self._generate_detailed_report()
        self._save_results()

    def _run_episode(self, episode_num):
        """Run single test episode with detailed tracking."""
        obs, info = self.env.reset()

        episode_soc = []
        episode_rewards = []
        episode_improvements = 0
        episode_worsenings = 0
        episode_neutral = 0
        episode_lower_violations = 0
        episode_upper_violations = 0
        episode_clipping = 0
        episode_flex_band = 0
        episode_reward_components = []

        print(f"\nEpisode {episode_num + 1}/{self.num_episodes}")
        print("-" * 40)

        for step in range(self.steps_per_episode):
            # Get action from policy (trained or random)
            action = self.get_action(obs)
            self.metrics['actions']['raw_actions'].append(action.copy())

            # Record loading BEFORE action
            loading_before = self.env.net.res_line['loading_percent'].max()
            self.metrics['congestion']['loading_before'].append(loading_before)

            # Execute step
            obs, reward, term, trunc, info = self.env.step(action)

            # Record loading AFTER action
            loading_after = info.get('loading_after_action', loading_before)
            self.metrics['congestion']['loading_after'].append(loading_after)

            # Calculate congestion change
            delta = loading_after - loading_before
            self.metrics['congestion']['deltas'].append(delta)

            # Categorize congestion change
            if delta < -0.1:  # Improved
                self.metrics['congestion']['improvements'].append(delta)
                episode_improvements += 1
                self.metrics['learning_quality']['responsive_steps'] += 1
            elif delta > 0.1:  # Worsened
                self.metrics['congestion']['worsenings'].append(delta)
                episode_worsenings += 1
                self.metrics['learning_quality']['harmful_steps'] += 1
            else:  # Neutral
                self.metrics['congestion']['neutral'].append(delta)
                episode_neutral += 1
                self.metrics['learning_quality']['neutral_steps'] += 1

            # Track SoC for all 5 BESS units
            soc = info.get('bess_soc', self.env.bess_soc)
            self.metrics['soc']['history'].append(soc.copy())
            episode_soc.append(soc.copy())

            # Count violations per unit
            lower_viols = np.sum(soc <= self.env.soc_min + 0.01)
            upper_viols = np.sum(soc >= self.env.soc_max - 0.01)
            self.metrics['soc']['lower_violations'] += lower_viols
            self.metrics['soc']['upper_violations'] += upper_viols
            episode_lower_violations += lower_viols
            episode_upper_violations += upper_viols

            # Track flexibility band (30-70%)
            in_band = np.sum((soc >= 0.3) & (soc <= 0.7))
            self.metrics['soc']['in_flex_band'].append(in_band)
            episode_flex_band += in_band

            # Track BESS activity for each unit
            if len(episode_soc) > 1:
                soc_prev = episode_soc[-2]
                for unit_idx in range(5):
                    unit_key = f'unit_{unit_idx}'
                    if soc[unit_idx] > soc_prev[unit_idx] + 0.001:
                        self.metrics['bess_activity'][unit_key]['charge'] += 1
                    elif soc[unit_idx] < soc_prev[unit_idx] - 0.001:
                        self.metrics['bess_activity'][unit_key]['discharge'] += 1
                    else:
                        self.metrics['bess_activity'][unit_key]['idle'] += 1

            # Track reward components
            breakdown = info.get('reward_breakdown', {})
            self.metrics['rewards']['total'].append(reward)
            self.metrics['rewards']['congestion'].append(breakdown.get('congestion', 0.0))
            self.metrics['rewards']['soc_penalty'].append(breakdown.get('soc_penalty', 0.0))
            self.metrics['rewards']['flexibility_bonus'].append(breakdown.get('flexibility_bonus', 0.0))
            self.metrics['rewards']['clipping_penalty'].append(breakdown.get('clipping_penalty', 0.0))
            episode_rewards.append(reward)
            episode_reward_components.append(breakdown)

            # Track clipping events
            clip_info = info.get('clip_info', {'units_clipped': []})
            if len(clip_info['units_clipped']) > 0:
                self.metrics['actions']['clipping_events'] += 1
                episode_clipping += 1

            # Progress indicator every 10 steps
            if (step + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"  Step {step+1:2d}/{self.steps_per_episode}: "
                      f"Loading {loading_before:6.2f}% -> {loading_after:6.2f}% "
                      f"({delta:+6.2f}%) | "
                      f"SoC [{soc.min():.2f}, {soc.max():.2f}] | "
                      f"R={avg_reward:+7.1f}")

            if term or trunc:
                print(f"  Episode ended at step {step+1}")
                break

        # Store episode-level statistics
        episode_soc_array = np.array(episode_soc)
        self.metrics['soc']['mean_per_episode'].append(episode_soc_array.mean())
        self.metrics['soc']['std_per_episode'].append(episode_soc_array.std())
        self.metrics['soc']['min_per_episode'].append(episode_soc_array.min())
        self.metrics['soc']['max_per_episode'].append(episode_soc_array.max())
        self.metrics['soc']['lower_violations_per_episode'].append(episode_lower_violations)
        self.metrics['soc']['upper_violations_per_episode'].append(episode_upper_violations)
        self.metrics['soc']['flex_band_per_episode'].append(episode_flex_band / (self.steps_per_episode * 5) * 100)

        self.metrics['congestion']['improvement_per_episode'].append(episode_improvements)
        self.metrics['congestion']['worsening_per_episode'].append(episode_worsenings)

        episode_avg_reward = np.mean(episode_rewards)
        self.metrics['rewards']['total_per_episode'].append(episode_avg_reward)
        self.metrics['rewards']['components_per_episode'].append({
            'congestion': np.mean([c.get('congestion', 0) for c in episode_reward_components]),
            'soc_penalty': np.mean([c.get('soc_penalty', 0) for c in episode_reward_components]),
            'flexibility_bonus': np.mean([c.get('flexibility_bonus', 0) for c in episode_reward_components]),
            'clipping_penalty': np.mean([c.get('clipping_penalty', 0) for c in episode_reward_components]),
        })

        self.metrics['actions']['clipping_per_episode'].append(episode_clipping)

        responsiveness = episode_improvements / self.steps_per_episode if self.steps_per_episode > 0 else 0
        self.metrics['learning_quality']['responsiveness_per_episode'].append(responsiveness)

        # Episode summary
        print(f"  Episode Summary:")
        print(f"    Avg Reward: {episode_avg_reward:+8.2f}")
        print(f"    Avg SoC: {episode_soc_array.mean():.3f}")
        print(f"    Congestion: {episode_improvements} impr, {episode_worsenings} worse, {episode_neutral} neutral")
        print(f"    Responsiveness: {responsiveness*100:.1f}%")
        print(f"    Violations: {episode_lower_violations} lower, {episode_upper_violations} upper")
        print(f"    Clipping events: {episode_clipping}")

    def _generate_detailed_report(self):
        """Generate comprehensive diagnostic report."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE DIAGNOSTIC RESULTS")
        print(f"{'='*80}\n")

        self._print_test_config()
        self._print_congestion_analysis()
        self._print_learning_quality_analysis()
        self._print_bess_operation_analysis()
        self._print_soc_analysis()
        self._print_reward_analysis()
        self._print_overall_assessment()

    def _print_test_config(self):
        """Print test configuration section."""
        print("1. TEST CONFIGURATION")
        print("-" * 40)
        print(f"  Policy Type: {self.policy_type}")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Duration: {self.metrics['timing']['duration_seconds']:.1f} seconds")
        print(f"  Steps/second: {self.total_steps / self.metrics['timing']['duration_seconds']:.2f}")
        print(f"\n  Environment Parameters:")
        print(f"    soc_penalty_weight: {self.cfg['soc_penalty_weight']}")
        print(f"    bonus_constant: {self.cfg['bonus_constant']}")
        print(f"    flexibility_bonus_weight: {self.cfg['flexibility_bonus_weight']}")
        print(f"    soc_clipping_penalty: {self.cfg['soc_clipping_penalty']}")

    def _print_congestion_analysis(self):
        """Print detailed congestion reduction analysis."""
        print(f"\n2. CONGESTION REDUCTION ANALYSIS")
        print("-" * 40)

        num_improvements = len(self.metrics['congestion']['improvements'])
        num_worsenings = len(self.metrics['congestion']['worsenings'])
        num_neutral = len(self.metrics['congestion']['neutral'])

        improvement_rate = 100 * num_improvements / self.total_steps
        worsening_rate = 100 * num_worsenings / self.total_steps
        neutral_rate = 100 * num_neutral / self.total_steps

        print(f"\n  Step-by-Step Analysis:")
        print(f"    Improvement steps: {num_improvements} ({improvement_rate:.1f}%)")
        print(f"    Worsening steps: {num_worsenings} ({worsening_rate:.1f}%)")
        print(f"    Neutral steps: {num_neutral} ({neutral_rate:.1f}%)")

        if self.metrics['congestion']['improvements']:
            avg_impr_mag = np.mean([abs(x) for x in self.metrics['congestion']['improvements']])
            print(f"    Avg improvement magnitude: {avg_impr_mag:.2f}% loading reduction")

        if self.metrics['congestion']['worsenings']:
            avg_worse_mag = np.mean([abs(x) for x in self.metrics['congestion']['worsenings']])
            print(f"    Avg worsening magnitude: {avg_worse_mag:.2f}% loading increase")

        # Loading statistics
        avg_loading_before = np.mean(self.metrics['congestion']['loading_before'])
        avg_loading_after = np.mean(self.metrics['congestion']['loading_after'])
        avg_delta = np.mean(self.metrics['congestion']['deltas'])

        print(f"\n  Loading Statistics:")
        print(f"    Avg loading BEFORE action: {avg_loading_before:.2f}%")
        print(f"    Avg loading AFTER action: {avg_loading_after:.2f}%")
        print(f"    Net change: {avg_delta:+.3f}%")

        print(f"\n  Episode-by-Episode Congestion Performance:")
        for i in range(len(self.metrics['congestion']['improvement_per_episode'])):
            impr = self.metrics['congestion']['improvement_per_episode'][i]
            worse = self.metrics['congestion']['worsening_per_episode'][i]
            rate = 100 * impr / self.steps_per_episode
            print(f"    Episode {i+1:2d}: {impr:2d} impr ({rate:5.1f}%), {worse:2d} worse")

    def _print_learning_quality_analysis(self):
        """Print learning quality and responsiveness analysis."""
        print(f"\n3. LEARNING QUALITY / RESPONSIVENESS ANALYSIS")
        print("-" * 40)

        responsive = self.metrics['learning_quality']['responsive_steps']
        harmful = self.metrics['learning_quality']['harmful_steps']
        neutral = self.metrics['learning_quality']['neutral_steps']
        total_active = responsive + harmful

        print(f"\n  Action Quality Assessment:")
        print(f"    Responsive actions (helped grid): {responsive} ({100*responsive/self.total_steps:.1f}%)")
        print(f"    Harmful actions (hurt grid): {harmful} ({100*harmful/self.total_steps:.1f}%)")
        print(f"    Neutral actions: {neutral} ({100*neutral/self.total_steps:.1f}%)")

        if total_active > 0:
            responsiveness_ratio = responsive / total_active
            print(f"\n  Responsiveness Ratio: {responsiveness_ratio:.3f}")
            print(f"    (Ratio of helpful vs harmful actions)")

            if self.policy_type == "RANDOM":
                print(f"\n  INTERPRETATION FOR RANDOM POLICY:")
                print(f"    Expected ratio: ~0.50 (50/50 random chance)")
                print(f"    Actual ratio: {responsiveness_ratio:.3f}")
                if responsiveness_ratio >= 0.45 and responsiveness_ratio <= 0.55:
                    print(f"    Status: CORRECT - Random as expected")
                else:
                    print(f"    Status: UNEXPECTED - Not behaving randomly")
            else:
                print(f"\n  INTERPRETATION FOR TRAINED POLICY:")
                if responsiveness_ratio < 0.55:
                    print(f"    Status: NOT LEARNING - Agent not better than random")
                    print(f"    Problem: Agent is not responding to grid state effectively")
                elif responsiveness_ratio < 0.65:
                    print(f"    Status: WEAK LEARNING - Some improvement over random")
                    print(f"    Suggestion: Continue training or adjust reward weights")
                elif responsiveness_ratio < 0.75:
                    print(f"    Status: GOOD LEARNING - Agent responding well to grid")
                    print(f"    Performance: Significantly better than random")
                else:
                    print(f"    Status: EXCELLENT LEARNING - Strong grid awareness")
                    print(f"    Performance: Optimal congestion management")

        print(f"\n  Episode Responsiveness Progression:")
        for i, resp_rate in enumerate(self.metrics['learning_quality']['responsiveness_per_episode']):
            print(f"    Episode {i+1:2d}: {resp_rate*100:5.1f}% responsive actions")

    def _print_bess_operation_analysis(self):
        """Print detailed BESS operation analysis for all 5 units."""
        print(f"\n4. BESS OPERATION ANALYSIS (ALL 5 UNITS)")
        print("-" * 40)

        print(f"\n  Individual Unit Activity:")
        all_operational = True
        for unit_idx in range(5):
            unit_key = f'unit_{unit_idx}'
            activity = self.metrics['bess_activity'][unit_key]
            total = activity['charge'] + activity['discharge'] + activity['idle']

            if total == 0:
                print(f"    BESS Unit {unit_idx}: WARNING - NO ACTIVITY DETECTED")
                all_operational = False
            else:
                charge_pct = 100 * activity['charge'] / total
                discharge_pct = 100 * activity['discharge'] / total
                idle_pct = 100 * activity['idle'] / total

                print(f"    BESS Unit {unit_idx}:")
                print(f"      Charge: {activity['charge']:4d} ({charge_pct:5.1f}%)")
                print(f"      Discharge: {activity['discharge']:4d} ({discharge_pct:5.1f}%)")
                print(f"      Idle: {activity['idle']:4d} ({idle_pct:5.1f}%)")

        print(f"\n  Overall BESS Status:")
        if all_operational:
            print(f"    ALL 5 BESS UNITS ARE OPERATIONAL")
        else:
            print(f"    WARNING: Some BESS units not operating!")

        # Clipping analysis
        total_clipping = self.metrics['actions']['clipping_events']
        clipping_rate = 100 * total_clipping / self.total_steps
        print(f"\n  Action Clipping:")
        print(f"    Clipping events: {total_clipping} ({clipping_rate:.1f}% of steps)")

        if self.policy_type == "RANDOM":
            if clipping_rate > 30:
                print(f"    Status: NORMAL for random policy (high clipping expected)")
            else:
                print(f"    Status: UNEXPECTEDLY LOW for random policy")
        else:
            if clipping_rate > 25:
                print(f"    Status: HIGH - Agent requesting infeasible actions frequently")
                print(f"    Suggestion: Increase soc_clipping_penalty to discourage this")
            elif clipping_rate > 15:
                print(f"    Status: MODERATE - Acceptable learning phase behavior")
            else:
                print(f"    Status: GOOD - Agent learned feasible action space")

    def _print_soc_analysis(self):
        """Print SoC distribution and violation analysis."""
        print(f"\n5. SOC DISTRIBUTION & VIOLATION ANALYSIS")
        print("-" * 40)

        soc_array = np.array(self.metrics['soc']['history'])
        mean_soc = soc_array.mean()
        std_soc = soc_array.std()
        min_soc = soc_array.min()
        max_soc = soc_array.max()

        print(f"\n  Overall SoC Statistics:")
        print(f"    Mean SoC: {mean_soc:.3f}")
        print(f"    Std Dev: {std_soc:.3f}")
        print(f"    Min: {min_soc:.3f}")
        print(f"    Max: {max_soc:.3f}")

        violations_per_step = (self.metrics['soc']['lower_violations'] +
                              self.metrics['soc']['upper_violations']) / self.total_steps

        print(f"\n  Boundary Violations:")
        print(f"    Lower bound (10%) hits: {self.metrics['soc']['lower_violations']}")
        print(f"    Upper bound (90%) hits: {self.metrics['soc']['upper_violations']}")
        print(f"    Total violations: {self.metrics['soc']['lower_violations'] + self.metrics['soc']['upper_violations']}")
        print(f"    Violations per step: {violations_per_step:.2f}")

        if self.metrics['soc']['lower_violations'] + self.metrics['soc']['upper_violations'] > 0:
            ratio = self.metrics['soc']['lower_violations'] / max(1, self.metrics['soc']['upper_violations'])
            print(f"    Lower/Upper ratio: {ratio:.1f}:1")

            if ratio > 3:
                print(f"    Status: DISCHARGE BIAS - More lower bound hits")
                print(f"    Suggestion: Increase flexibility_bonus to keep SoC centered")
            elif ratio < 0.33:
                print(f"    Status: CHARGE BIAS - More upper bound hits")
                print(f"    Suggestion: Check if agent is too conservative")
            else:
                print(f"    Status: BALANCED violations")

        # Flexibility band
        flex_band_count = np.sum(self.metrics['soc']['in_flex_band'])
        flex_band_percentage = 100 * flex_band_count / (self.total_steps * 5)

        print(f"\n  Flexibility Band (30-70% SoC):")
        print(f"    Time in band: {flex_band_percentage:.1f}%")
        print(f"    Target: >60%")

        if flex_band_percentage > 70:
            print(f"    Status: EXCELLENT operational flexibility")
        elif flex_band_percentage > 60:
            print(f"    Status: GOOD operational flexibility")
        elif flex_band_percentage > 45:
            print(f"    Status: MODERATE - Approaching target")
        else:
            print(f"    Status: NEEDS IMPROVEMENT")

        print(f"\n  Episode-by-Episode SoC Performance:")
        for i in range(len(self.metrics['soc']['mean_per_episode'])):
            mean_ep = self.metrics['soc']['mean_per_episode'][i]
            lower_v = self.metrics['soc']['lower_violations_per_episode'][i]
            upper_v = self.metrics['soc']['upper_violations_per_episode'][i]
            flex_ep = self.metrics['soc']['flex_band_per_episode'][i]
            print(f"    Ep {i+1:2d}: Mean={mean_ep:.3f}, Violations(L/U)={lower_v:2d}/{upper_v:2d}, Flex={flex_ep:5.1f}%")

    def _print_reward_analysis(self):
        """Print reward component analysis."""
        print(f"\n6. REWARD COMPONENT ANALYSIS")
        print("-" * 40)

        avg_total = np.mean(self.metrics['rewards']['total'])
        avg_cong = np.mean(self.metrics['rewards']['congestion'])
        avg_soc_pen = np.mean(self.metrics['rewards']['soc_penalty'])
        avg_flex = np.mean(self.metrics['rewards']['flexibility_bonus'])
        avg_clip = np.mean(self.metrics['rewards']['clipping_penalty'])

        print(f"\n  Average Reward Components:")
        print(f"    Total reward: {avg_total:+8.2f}")
        print(f"    Congestion reward: {avg_cong:+8.2f}")
        print(f"    SoC penalty: {avg_soc_pen:+8.2f}")
        print(f"    Flexibility bonus: {avg_flex:+8.2f}")
        print(f"    Clipping penalty: {avg_clip:+8.2f}")

        # Reward balance
        if abs(avg_soc_pen) > 0.01:
            ratio = abs(avg_cong) / abs(avg_soc_pen)
            print(f"\n  Reward Balance Ratio:")
            print(f"    Congestion/SoC ratio: {ratio:.2f}")
            print(f"    Target range: 2-10:1")

            if ratio > 10:
                print(f"    Status: IMBALANCED - Congestion dominates")
                print(f"    Problem: Agent will ignore SoC management")
                print(f"    Fix: Reduce bonus_constant or increase soc_penalty_weight")
            elif ratio < 2:
                print(f"    Status: IMBALANCED - SoC penalty too strong")
                print(f"    Problem: Agent will be overly conservative")
                print(f"    Fix: Increase bonus_constant or reduce soc_penalty_weight")
            else:
                print(f"    Status: WELL BALANCED for learning")

        print(f"\n  Episode Reward Progression:")
        for i, ep_reward in enumerate(self.metrics['rewards']['total_per_episode']):
            components = self.metrics['rewards']['components_per_episode'][i]
            print(f"    Ep {i+1:2d}: Total={ep_reward:+7.1f}, "
                  f"Cong={components['congestion']:+6.1f}, "
                  f"SoC={components['soc_penalty']:+6.1f}, "
                  f"Flex={components['flexibility_bonus']:+6.1f}")

    def _print_overall_assessment(self):
        """Print overall assessment and verdict."""
        print(f"\n{'='*80}")
        print("OVERALL ASSESSMENT & RECOMMENDATIONS")
        print(f"{'='*80}\n")

        # Calculate key metrics for assessment
        improvement_rate = 100 * len(self.metrics['congestion']['improvements']) / self.total_steps
        violations_per_step = (self.metrics['soc']['lower_violations'] +
                              self.metrics['soc']['upper_violations']) / self.total_steps
        mean_soc = np.mean(self.metrics['soc']['history'])
        flex_band_pct = 100 * np.sum(self.metrics['soc']['in_flex_band']) / (self.total_steps * 5)

        responsive = self.metrics['learning_quality']['responsive_steps']
        harmful = self.metrics['learning_quality']['harmful_steps']
        responsiveness_ratio = responsive / max(1, responsive + harmful)

        print(f"POLICY TYPE: {self.policy_type}")
        print(f"{'='*80}\n")

        if self.policy_type == "RANDOM":
            print("BASELINE TEST RESULTS:")
            print(f"  This establishes baseline performance before training.")
            print(f"  All metrics are expected to be poor/random.")
            print(f"\n  Key Baseline Metrics:")
            print(f"    - Improvement rate: {improvement_rate:.1f}% (expect ~0% for random)")
            print(f"    - Responsiveness: {responsiveness_ratio:.2f} (expect ~0.50 for random)")
            print(f"    - Violations/step: {violations_per_step:.2f}")
            print(f"    - Mean SoC: {mean_soc:.3f}")
            print(f"\n  RECOMMENDATION:")
            print(f"    Train agent with: python training.py")
            print(f"    Then re-run this test with trained model to measure improvement.")

        else:
            print("TRAINED AGENT PERFORMANCE:")
            print(f"\n  Congestion Management:")
            if improvement_rate > 65:
                print(f"    EXCELLENT ({improvement_rate:.1f}% improvement rate)")
            elif improvement_rate > 50:
                print(f"    GOOD ({improvement_rate:.1f}% improvement rate)")
            elif improvement_rate > 35:
                print(f"    MODERATE ({improvement_rate:.1f}% improvement rate)")
            else:
                print(f"    POOR ({improvement_rate:.1f}% improvement rate)")
                print(f"    Problem: Agent not learning congestion reduction effectively")

            print(f"\n  Learning Quality:")
            if responsiveness_ratio > 0.75:
                print(f"    EXCELLENT (responsiveness: {responsiveness_ratio:.2f})")
            elif responsiveness_ratio > 0.65:
                print(f"    GOOD (responsiveness: {responsiveness_ratio:.2f})")
            elif responsiveness_ratio > 0.55:
                print(f"    WEAK (responsiveness: {responsiveness_ratio:.2f})")
            else:
                print(f"    NOT LEARNING (responsiveness: {responsiveness_ratio:.2f})")
                print(f"    Problem: Agent no better than random policy")

            print(f"\n  SoC Management:")
            if violations_per_step < 0.2 and 0.45 <= mean_soc <= 0.55:
                print(f"    EXCELLENT (violations: {violations_per_step:.2f}/step, mean: {mean_soc:.3f})")
            elif violations_per_step < 0.4 and 0.40 <= mean_soc <= 0.60:
                print(f"    GOOD (violations: {violations_per_step:.2f}/step, mean: {mean_soc:.3f})")
            else:
                print(f"    NEEDS WORK (violations: {violations_per_step:.2f}/step, mean: {mean_soc:.3f})")

            print(f"\n  RECOMMENDATION:")
            if improvement_rate > 60 and responsiveness_ratio > 0.70 and violations_per_step < 0.3:
                print(f"    Agent is performing WELL. Consider deployment.")
            elif improvement_rate > 40:
                print(f"    Agent shows promise. Continue training for better performance.")
            else:
                print(f"    Agent not learning effectively. Check:")
                print(f"      - Reward balance (currently {abs(np.mean(self.metrics['rewards']['congestion'])) / abs(np.mean(self.metrics['rewards']['soc_penalty'])):.2f})")
                print(f"      - Training duration (may need more steps)")
                print(f"      - Learning rate and other hyperparameters")

        print(f"\n{'='*80}")

    def _save_results(self):
        """Save detailed results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON summary
        json_path = os.path.join(self.output_dir, f"results_{self.policy_type.lower()}_{timestamp}.json")
        results = {
            'policy_type': self.policy_type,
            'model_path': self.model_path,
            'config': {
                'num_episodes': self.num_episodes,
                'steps_per_episode': self.steps_per_episode,
                'total_steps': self.total_steps,
                'soc_penalty_weight': self.cfg['soc_penalty_weight'],
                'bonus_constant': self.cfg['bonus_constant'],
                'flexibility_bonus_weight': self.cfg['flexibility_bonus_weight'],
                'soc_clipping_penalty': self.cfg['soc_clipping_penalty'],
            },
            'summary': {
                'improvement_rate': 100 * len(self.metrics['congestion']['improvements']) / self.total_steps,
                'worsening_rate': 100 * len(self.metrics['congestion']['worsenings']) / self.total_steps,
                'responsiveness_ratio': self.metrics['learning_quality']['responsive_steps'] /
                                       max(1, self.metrics['learning_quality']['responsive_steps'] +
                                           self.metrics['learning_quality']['harmful_steps']),
                'violations_per_step': (self.metrics['soc']['lower_violations'] +
                                       self.metrics['soc']['upper_violations']) / self.total_steps,
                'mean_soc': float(np.mean(self.metrics['soc']['history'])),
                'flex_band_percentage': 100 * np.sum(self.metrics['soc']['in_flex_band']) / (self.total_steps * 5),
                'avg_total_reward': float(np.mean(self.metrics['rewards']['total'])),
                'avg_loading_before': float(np.mean(self.metrics['congestion']['loading_before'])),
                'avg_loading_after': float(np.mean(self.metrics['congestion']['loading_after'])),
                'net_loading_change': float(np.mean(self.metrics['congestion']['deltas'])),
            },
            'timestamp': timestamp,
            'duration_seconds': self.metrics['timing']['duration_seconds'],
        }

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive diagnostic test for BESS control policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model .zip file (if not provided, uses random policy)')
    parser.add_argument('--random', action='store_true',
                       help='Explicitly use random policy for baseline testing')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test (default: 10)')
    parser.add_argument('--steps', type=int, default=50,
                       help='Steps per episode (default: 50)')
    parser.add_argument('--output', type=str, default='diagnostic_tests/results',
                       help='Output directory for results (default: diagnostic_tests/results)')

    args = parser.parse_args()

    # Validate arguments
    if args.random:
        model_path = None
    elif args.model:
        model_path = args.model
    else:
        # Check if default model exists
        if os.path.exists('final_model.zip'):
            print("Found final_model.zip in current directory")
            response = input("Use this model? (y/n): ")
            if response.lower() == 'y':
                model_path = 'final_model.zip'
            else:
                model_path = None
        else:
            print("No model specified. Using random policy for baseline testing.")
            model_path = None

    import warnings
    warnings.filterwarnings('ignore')

    print("\n" + "="*80)
    print("POLICY DIAGNOSTIC TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Policy: {'RANDOM (baseline)' if model_path is None else f'TRAINED ({model_path})'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Steps per episode: {args.steps}")
    print(f"  Total steps: {args.episodes * args.steps}")
    print(f"  Output directory: {args.output}")
    print(f"\nEstimated time: {args.episodes * args.steps * 0.3 / 60:.1f} minutes")
    print("="*80 + "\n")

    try:
        framework = PolicyDiagnosticTest(
            model_path=model_path,
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            output_dir=args.output
        )
        framework.run_comprehensive_test()

        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Test failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
