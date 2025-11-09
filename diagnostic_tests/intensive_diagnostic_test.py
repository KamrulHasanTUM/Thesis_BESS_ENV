"""
INTENSIVE DIAGNOSTIC TEST FRAMEWORK
====================================
Comprehensive, reusable test suite for BESS environment configuration validation.

This framework can be used for any future diagnostic testing without modification.
Just run: python intensive_diagnostic_test.py

Features:
- Configurable test duration (episodes and steps)
- Detailed before/after comparison
- Learning progress tracking
- Multiple test scenarios (random, trained agent, specific patterns)
- Complete metrics tracking and visualization
- Automatic report generation
"""

import numpy as np
from config import load_config, create_bess_env_config
from ENV_BESS_main import ENV_BESS
import sys
import time
from datetime import datetime
import json

class DiagnosticTestFramework:
    """Comprehensive diagnostic test framework for BESS environment."""

    def __init__(self, num_episodes=15, steps_per_episode=50):
        """
        Initialize test framework.

        Args:
            num_episodes: Number of episodes to test
            steps_per_episode: Steps per episode
        """
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.total_steps = num_episodes * steps_per_episode

        # Load configuration
        self.cfg = create_bess_env_config(load_config())
        self.env = None

        # Metrics storage
        self.metrics = {
            'congestion': {
                'improvements': [],
                'worsenings': [],
                'neutral': [],
                'loading_before': [],
                'loading_after': [],
                'deltas': [],
                'improvement_magnitudes': [],
                'worsening_magnitudes': [],
            },
            'soc': {
                'history': [],
                'mean_per_episode': [],
                'std_per_episode': [],
                'lower_violations': 0,
                'upper_violations': 0,
                'in_flex_band': [],
                'lower_violations_per_episode': [],
                'upper_violations_per_episode': [],
            },
            'rewards': {
                'total': [],
                'congestion': [],
                'soc_penalty': [],
                'flexibility_bonus': [],
                'clipping_penalty': [],
                'total_per_episode': [],
            },
            'actions': {
                'charge_count': 0,
                'discharge_count': 0,
                'idle_count': 0,
                'clipping_events': 0,
                'charge_per_episode': [],
                'discharge_per_episode': [],
            },
            'learning': {
                'reward_moving_avg': [],
                'improvement_rate_moving_avg': [],
                'episode_trends': [],
            },
            'timing': {
                'start_time': None,
                'end_time': None,
                'duration_seconds': 0,
            }
        }

    def run_comprehensive_test(self):
        """Run complete diagnostic test suite."""
        print("="*80)
        print("INTENSIVE DIAGNOSTIC TEST FRAMEWORK")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Test Episodes: {self.num_episodes}")
        print(f"  Steps per Episode: {self.steps_per_episode}")
        print(f"  Total Steps: {self.total_steps}")
        print(f"\nEnvironment Parameters:")
        print(f"  soc_penalty_weight: {self.cfg['soc_penalty_weight']}")
        print(f"  bonus_constant: {self.cfg['bonus_constant']}")
        print(f"  max_step: {self.cfg['max_step']}")
        print(f"  flexibility_bonus_weight: {self.cfg['flexibility_bonus_weight']}")
        print(f"  soc_clipping_penalty: {self.cfg['soc_clipping_penalty']}")
        print(f"  soc_boundary_margin: {self.cfg['soc_boundary_margin']}")

        self.metrics['timing']['start_time'] = datetime.now()

        # Initialize environment
        self.env = ENV_BESS(**self.cfg)

        print(f"\n{'='*80}")
        print(f"Starting Test Run...")
        print(f"{'='*80}\n")

        # Run test episodes
        for episode in range(self.num_episodes):
            self._run_episode(episode)

        self.metrics['timing']['end_time'] = datetime.now()
        self.metrics['timing']['duration_seconds'] = (
            self.metrics['timing']['end_time'] - self.metrics['timing']['start_time']
        ).total_seconds()

        # Generate report
        self._generate_report()

    def _run_episode(self, episode_num):
        """Run a single test episode."""
        obs, info = self.env.reset()
        episode_soc = []
        episode_rewards = []
        episode_improvements = 0
        episode_worsenings = 0
        episode_lower_violations = 0
        episode_upper_violations = 0
        episode_charges = 0
        episode_discharges = 0
        episode_improvement_magnitudes = []
        episode_worsening_magnitudes = []

        print(f"\nEpisode {episode_num + 1}/{self.num_episodes}")
        print("-" * 40)

        for step in range(self.steps_per_episode):
            # Random policy for baseline testing
            # Can be replaced with trained agent for validation
            action = self._generate_test_action(step)

            # Record loading before
            loading_before = self.env.net.res_line['loading_percent'].max()
            self.metrics['congestion']['loading_before'].append(loading_before)

            # Execute step
            obs, reward, term, trunc, info = self.env.step(action)

            # Record loading after
            loading_after = info.get('loading_after_action', loading_before)
            self.metrics['congestion']['loading_after'].append(loading_after)

            # Calculate delta
            delta = loading_after - loading_before
            self.metrics['congestion']['deltas'].append(delta)

            # Categorize step and track episode counts
            if delta < -0.1:
                self.metrics['congestion']['improvements'].append(delta)
                self.metrics['congestion']['improvement_magnitudes'].append(abs(delta))
                episode_improvements += 1
                episode_improvement_magnitudes.append(abs(delta))
            elif delta > 0.1:
                self.metrics['congestion']['worsenings'].append(delta)
                self.metrics['congestion']['worsening_magnitudes'].append(abs(delta))
                episode_worsenings += 1
                episode_worsening_magnitudes.append(abs(delta))
            else:
                self.metrics['congestion']['neutral'].append(delta)

            # Track SoC
            soc = info.get('bess_soc', self.env.bess_soc)
            self.metrics['soc']['history'].append(soc.copy())
            episode_soc.append(soc.copy())

            # Count violations (per-step and per-episode)
            lower_viols = np.sum(soc <= self.env.soc_min + 0.01)
            upper_viols = np.sum(soc >= self.env.soc_max - 0.01)
            self.metrics['soc']['lower_violations'] += lower_viols
            self.metrics['soc']['upper_violations'] += upper_viols
            episode_lower_violations += lower_viols
            episode_upper_violations += upper_viols

            # Track if in flexibility band
            in_band = np.sum((soc >= 0.3) & (soc <= 0.7))
            self.metrics['soc']['in_flex_band'].append(in_band)

            # Track rewards
            breakdown = info.get('reward_breakdown', {})
            self.metrics['rewards']['total'].append(reward)
            self.metrics['rewards']['congestion'].append(breakdown.get('congestion', 0.0))
            self.metrics['rewards']['soc_penalty'].append(breakdown.get('soc_penalty', 0.0))
            self.metrics['rewards']['flexibility_bonus'].append(breakdown.get('flexibility_bonus', 0.0))
            self.metrics['rewards']['clipping_penalty'].append(breakdown.get('clipping_penalty', 0.0))

            episode_rewards.append(reward)

            # Track actions
            clip_info = info.get('clip_info', {'units_clipped': []})
            if len(clip_info['units_clipped']) > 0:
                self.metrics['actions']['clipping_events'] += 1

            # Count charge/discharge per BESS unit
            for i, soc_current in enumerate(soc):
                if len(episode_soc) > 1:
                    soc_prev = episode_soc[-2][i]
                    if soc_current > soc_prev + 0.001:
                        self.metrics['actions']['charge_count'] += 1
                        episode_charges += 1
                    elif soc_current < soc_prev - 0.001:
                        self.metrics['actions']['discharge_count'] += 1
                        episode_discharges += 1
                    else:
                        self.metrics['actions']['idle_count'] += 1

            # Progress indicator
            if (step + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"  Step {step+1:2d}/{self.steps_per_episode}: "
                      f"Loading {loading_before:6.2f}% -> {loading_after:6.2f}% "
                      f"({delta:+6.2f}%) | "
                      f"SoC [{soc.min():.2f}, {soc.max():.2f}] | "
                      f"R_avg={avg_reward:+7.1f}")

            if term or trunc:
                print(f"  Episode ended at step {step+1}")
                break

        # Episode summary statistics
        episode_soc_array = np.array(episode_soc)
        self.metrics['soc']['mean_per_episode'].append(episode_soc_array.mean())
        self.metrics['soc']['std_per_episode'].append(episode_soc_array.std())
        self.metrics['soc']['lower_violations_per_episode'].append(episode_lower_violations)
        self.metrics['soc']['upper_violations_per_episode'].append(episode_upper_violations)

        episode_avg_reward = np.mean(episode_rewards)
        self.metrics['rewards']['total_per_episode'].append(episode_avg_reward)
        self.metrics['actions']['charge_per_episode'].append(episode_charges)
        self.metrics['actions']['discharge_per_episode'].append(episode_discharges)

        # Calculate learning indicators
        window_size = min(3, episode_num + 1)
        if len(self.metrics['rewards']['total_per_episode']) >= window_size:
            moving_avg = np.mean(self.metrics['rewards']['total_per_episode'][-window_size:])
            self.metrics['learning']['reward_moving_avg'].append(moving_avg)

        # Calculate improvement rate moving average
        if episode_num > 0:
            recent_impr_rate = episode_improvements / self.steps_per_episode
            self.metrics['learning']['improvement_rate_moving_avg'].append(recent_impr_rate)

        # Episode trend assessment
        trend_status = "STABLE"
        if episode_num > 0:
            prev_reward = self.metrics['rewards']['total_per_episode'][-2] if len(self.metrics['rewards']['total_per_episode']) > 1 else 0
            if episode_avg_reward > prev_reward + 5:
                trend_status = "IMPROVING"
            elif episode_avg_reward < prev_reward - 5:
                trend_status = "DECLINING"
        self.metrics['learning']['episode_trends'].append(trend_status)

        # Enhanced episode summary
        avg_impr_mag = np.mean(episode_improvement_magnitudes) if episode_improvement_magnitudes else 0
        avg_wors_mag = np.mean(episode_worsening_magnitudes) if episode_worsening_magnitudes else 0

        print(f"  Episode Summary: Avg Reward = {episode_avg_reward:+8.2f}, "
              f"Avg SoC = {episode_soc_array.mean():.3f}")
        print(f"    Congestion: {episode_improvements} impr ({avg_impr_mag:.2f}% avg), "
              f"{episode_worsenings} worse ({avg_wors_mag:.2f}% avg)")
        print(f"    Violations: {episode_lower_violations} lower, {episode_upper_violations} upper")
        print(f"    Trend: {trend_status}")

    def _generate_test_action(self, step):
        """
        Generate test action.

        Can be modified to test different scenarios:
        - Random policy (default)
        - Trained agent
        - Specific patterns
        """
        # Random policy for baseline
        return np.random.uniform(-0.5, 0.5, size=5).astype(np.float32)

    def _generate_report(self):
        """Generate comprehensive diagnostic report."""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC TEST RESULTS")
        print(f"{'='*80}\n")

        # Test configuration summary
        print("1. TEST CONFIGURATION")
        print("-" * 40)
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Duration: {self.metrics['timing']['duration_seconds']:.1f} seconds")
        print(f"  Steps/second: {self.total_steps / self.metrics['timing']['duration_seconds']:.2f}")

        # Configuration parameters
        print(f"\n2. ENVIRONMENT PARAMETERS")
        print("-" * 40)
        print(f"  soc_penalty_weight: {self.cfg['soc_penalty_weight']}")
        print(f"  bonus_constant: {self.cfg['bonus_constant']}")
        print(f"  max_step: {self.cfg['max_step']}")
        print(f"  flexibility_bonus_weight: {self.cfg['flexibility_bonus_weight']}")
        print(f"  soc_clipping_penalty: {self.cfg['soc_clipping_penalty']}")

        # Congestion management performance
        print(f"\n3. CONGESTION MANAGEMENT PERFORMANCE")
        print("-" * 40)

        num_improvements = len(self.metrics['congestion']['improvements'])
        num_worsenings = len(self.metrics['congestion']['worsenings'])
        num_neutral = len(self.metrics['congestion']['neutral'])

        improvement_rate = 100 * num_improvements / self.total_steps
        worsening_rate = 100 * num_worsenings / self.total_steps
        neutral_rate = 100 * num_neutral / self.total_steps

        print(f"  Improvements: {num_improvements} ({improvement_rate:.1f}%%)")
        print(f"  Worsenings: {num_worsenings} ({worsening_rate:.1f}%%)")
        print(f"  Neutral: {num_neutral} ({neutral_rate:.1f}%%)")

        if self.metrics['congestion']['improvements']:
            avg_improvement = np.mean([abs(x) for x in self.metrics['congestion']['improvements']])
            print(f"  Avg improvement magnitude: {avg_improvement:.2f}%%")

        if self.metrics['congestion']['worsenings']:
            avg_worsening = np.mean([abs(x) for x in self.metrics['congestion']['worsenings']])
            print(f"  Avg worsening magnitude: {avg_worsening:.2f}%%")

        # Loading statistics
        avg_loading_before = np.mean(self.metrics['congestion']['loading_before'])
        avg_loading_after = np.mean(self.metrics['congestion']['loading_after'])
        avg_delta = np.mean(self.metrics['congestion']['deltas'])

        print(f"\n  Loading Statistics:")
        print(f"    Avg loading before: {avg_loading_before:.2f}%%")
        print(f"    Avg loading after: {avg_loading_after:.2f}%%")
        print(f"    Avg delta: {avg_delta:+.2f}%%")

        # SoC performance
        print(f"\n4. SOC PERFORMANCE")
        print("-" * 40)

        violations_per_step = (self.metrics['soc']['lower_violations'] +
                               self.metrics['soc']['upper_violations']) / self.total_steps

        print(f"  Lower bound (10%%) violations: {self.metrics['soc']['lower_violations']}")
        print(f"  Upper bound (90%%) violations: {self.metrics['soc']['upper_violations']}")
        print(f"  Total violations: {self.metrics['soc']['lower_violations'] + self.metrics['soc']['upper_violations']}")
        print(f"  Violations per step: {violations_per_step:.2f}")

        # SoC distribution
        soc_array = np.array(self.metrics['soc']['history'])
        mean_soc = soc_array.mean()
        std_soc = soc_array.std()
        min_soc = soc_array.min()
        max_soc = soc_array.max()

        print(f"\n  SoC Distribution:")
        print(f"    Mean: {mean_soc:.3f}")
        print(f"    Std: {std_soc:.3f}")
        print(f"    Min: {min_soc:.3f}")
        print(f"    Max: {max_soc:.3f}")

        # Flexibility band
        flex_band_count = np.sum(self.metrics['soc']['in_flex_band'])
        flex_band_percentage = 100 * flex_band_count / (self.total_steps * 5)  # 5 BESS units
        print(f"    Time in flexibility band (30-70%%): {flex_band_percentage:.1f}%%")

        # Episode-by-episode SoC trend with violations
        print(f"\n  Episode SoC Trend:")
        for i in range(len(self.metrics['soc']['mean_per_episode'])):
            mean_ep = self.metrics['soc']['mean_per_episode'][i]
            std_ep = self.metrics['soc']['std_per_episode'][i]
            lower_v = self.metrics['soc']['lower_violations_per_episode'][i]
            upper_v = self.metrics['soc']['upper_violations_per_episode'][i]
            print(f"    Episode {i+1:2d}: Mean={mean_ep:.3f}, Std={std_ep:.3f}, "
                  f"Violations (L/U): {lower_v}/{upper_v}")

        # Reward analysis
        print(f"\n5. REWARD COMPONENT ANALYSIS")
        print("-" * 40)

        avg_total_reward = np.mean(self.metrics['rewards']['total'])
        avg_congestion_reward = np.mean(self.metrics['rewards']['congestion'])
        avg_soc_penalty = np.mean(self.metrics['rewards']['soc_penalty'])
        avg_flex_bonus = np.mean(self.metrics['rewards']['flexibility_bonus'])
        avg_clip_penalty = np.mean(self.metrics['rewards']['clipping_penalty'])

        print(f"  Avg total reward: {avg_total_reward:+8.2f}")
        print(f"  Avg congestion reward: {avg_congestion_reward:+8.2f}")
        print(f"  Avg soc_penalty: {avg_soc_penalty:+8.2f}")
        print(f"  Avg flexibility_bonus: {avg_flex_bonus:+8.2f}")
        print(f"  Avg clipping_penalty: {avg_clip_penalty:+8.2f}")

        # Reward balance
        if abs(avg_soc_penalty) > 0.01:
            ratio = abs(avg_congestion_reward) / abs(avg_soc_penalty)
            print(f"\n  Reward Balance:")
            print(f"    Congestion/SoC ratio: {ratio:.2f}")
            if ratio > 10:
                print(f"    Status: WARNING - Congestion dominates (ratio > 10)")
            elif ratio < 2:
                print(f"    Status: WARNING - SoC penalty too strong (ratio < 2)")
            else:
                print(f"    Status: GOOD - Balanced (2 < ratio < 10)")

        # Action analysis
        print(f"\n6. ACTION ANALYSIS")
        print("-" * 40)

        total_actions = (self.metrics['actions']['charge_count'] +
                        self.metrics['actions']['discharge_count'] +
                        self.metrics['actions']['idle_count'])

        if total_actions > 0:
            charge_pct = 100 * self.metrics['actions']['charge_count'] / total_actions
            discharge_pct = 100 * self.metrics['actions']['discharge_count'] / total_actions
            idle_pct = 100 * self.metrics['actions']['idle_count'] / total_actions

            print(f"  Charge actions: {self.metrics['actions']['charge_count']} ({charge_pct:.1f}%%)")
            print(f"  Discharge actions: {self.metrics['actions']['discharge_count']} ({discharge_pct:.1f}%%)")
            print(f"  Idle actions: {self.metrics['actions']['idle_count']} ({idle_pct:.1f}%%)")

            print(f"\n  Action Balance:")
            if 40 <= charge_pct <= 60:
                print(f"    Status: GOOD - Balanced charge/discharge")
            elif charge_pct < 30:
                print(f"    Status: WARNING - Insufficient charging (<30%%)")
            elif charge_pct > 70:
                print(f"    Status: WARNING - Excessive charging (>70%%)")
            else:
                print(f"    Status: MODERATE - Acceptable balance")

        # Clipping events
        clipping_rate = 100 * self.metrics['actions']['clipping_events'] / self.total_steps
        print(f"\n  Clipping Events: {self.metrics['actions']['clipping_events']} ({clipping_rate:.1f}%% of steps)")

        # Learning analysis
        print(f"\n7. LEARNING TREND ANALYSIS")
        print("-" * 40)

        if len(self.metrics['rewards']['total_per_episode']) > 0:
            print(f"\n  Episode-by-Episode Performance:")
            for i in range(len(self.metrics['rewards']['total_per_episode'])):
                ep_reward = self.metrics['rewards']['total_per_episode'][i]
                ep_trend = self.metrics['learning']['episode_trends'][i]
                ep_charges = self.metrics['actions']['charge_per_episode'][i]
                ep_discharges = self.metrics['actions']['discharge_per_episode'][i]

                # Get moving average if available
                moving_avg_str = ""
                if i < len(self.metrics['learning']['reward_moving_avg']):
                    moving_avg = self.metrics['learning']['reward_moving_avg'][i]
                    moving_avg_str = f", MA(3)={moving_avg:+7.1f}"

                print(f"    Ep {i+1:2d}: Reward={ep_reward:+7.1f}{moving_avg_str}, "
                      f"Trend={ep_trend:9s}, C/D={ep_charges}/{ep_discharges}")

            # Overall learning assessment
            if len(self.metrics['rewards']['total_per_episode']) >= 3:
                first_third = self.metrics['rewards']['total_per_episode'][:len(self.metrics['rewards']['total_per_episode'])//3]
                last_third = self.metrics['rewards']['total_per_episode'][-len(self.metrics['rewards']['total_per_episode'])//3:]

                if len(first_third) > 0 and len(last_third) > 0:
                    early_avg = np.mean(first_third)
                    late_avg = np.mean(last_third)
                    change = late_avg - early_avg

                    print(f"\n  Overall Learning Trajectory:")
                    print(f"    Early episodes avg: {early_avg:+7.1f}")
                    print(f"    Late episodes avg: {late_avg:+7.1f}")
                    print(f"    Change: {change:+7.1f} ({100*change/abs(early_avg) if early_avg != 0 else 0:+.1f}%%)")

                    if change > 10:
                        print(f"    Status: IMPROVING - Positive learning trend detected")
                    elif change < -10:
                        print(f"    Status: DECLINING - Performance degrading")
                    else:
                        print(f"    Status: STABLE - No clear learning trend (expected for random policy)")

            # Action responsiveness check (blind vs responsive)
            print(f"\n  Action Responsiveness Assessment:")

            # Check if actions correlate with congestion states
            if len(self.metrics['congestion']['deltas']) > 10:
                improvements = len(self.metrics['congestion']['improvements'])
                worsenings = len(self.metrics['congestion']['worsenings'])
                total_active = improvements + worsenings

                if total_active > 0:
                    improvement_ratio = improvements / total_active
                    print(f"    Improvement ratio: {improvement_ratio:.2f} ({improvements}/{total_active})")

                    if improvement_ratio < 0.3:
                        print(f"    Status: LIKELY BLIND - Very low improvement rate")
                        print(f"            Actions appear random/not responsive to grid state")
                    elif improvement_ratio < 0.45:
                        print(f"    Status: MOSTLY BLIND - Low improvement rate")
                        print(f"            Some responsiveness but mostly random")
                    else:
                        print(f"    Status: RESPONSIVE - Actions showing grid awareness")
                        print(f"            (Note: Random policy expected to be ~50%%)")

                # Check consistency across episodes
                improving_count = self.metrics['learning']['episode_trends'].count('IMPROVING')
                declining_count = self.metrics['learning']['episode_trends'].count('DECLINING')

                print(f"\n    Episode consistency:")
                print(f"      Improving episodes: {improving_count}/{len(self.metrics['learning']['episode_trends'])}")
                print(f"      Declining episodes: {declining_count}/{len(self.metrics['learning']['episode_trends'])}")

                if improving_count > len(self.metrics['learning']['episode_trends']) * 0.6:
                    print(f"      Status: CONSISTENT IMPROVEMENT - Agent may be learning")
                elif declining_count > len(self.metrics['learning']['episode_trends']) * 0.6:
                    print(f"      Status: CONSISTENT DECLINE - Check for issues")
                else:
                    print(f"      Status: VARIABLE - Expected for random/untrained policy")

        # Overall assessment
        print(f"\n{'='*80}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*80}\n")

        status_items = []

        # Congestion
        if improvement_rate > 40:
            status_items.append("PASS: Good congestion improvement rate (>40%%)")
        elif improvement_rate > 25:
            status_items.append("MODERATE: Fair congestion improvement rate (25-40%%)")
        else:
            status_items.append("NEEDS IMPROVEMENT: Low congestion improvement rate (<25%%)")

        # Violations
        if violations_per_step < 0.5:
            status_items.append("PASS: Low SoC boundary violations (<0.5 per step)")
        elif violations_per_step < 1.0:
            status_items.append("MODERATE: Moderate SoC violations (0.5-1.0 per step)")
        else:
            status_items.append("NEEDS IMPROVEMENT: High SoC violations (>1.0 per step)")

        # Flexibility
        if flex_band_percentage > 60:
            status_items.append("PASS: Excellent flexibility (>60%% in band)")
        elif flex_band_percentage > 45:
            status_items.append("MODERATE: Fair flexibility (45-60%% in band)")
        else:
            status_items.append("NEEDS IMPROVEMENT: Poor flexibility (<45%% in band)")

        # Reward balance
        if abs(avg_soc_penalty) > 0.01:
            ratio = abs(avg_congestion_reward) / abs(avg_soc_penalty)
            if 2 <= ratio <= 10:
                status_items.append("PASS: Well-balanced rewards (ratio 2-10)")
            else:
                status_items.append("NEEDS ADJUSTMENT: Reward imbalance detected")

        # Mean SoC
        if 0.45 <= mean_soc <= 0.55:
            status_items.append("PASS: Optimal mean SoC (45-55%%)")
        elif 0.40 <= mean_soc <= 0.60:
            status_items.append("MODERATE: Acceptable mean SoC (40-60%%)")
        else:
            status_items.append("NEEDS IMPROVEMENT: Mean SoC outside target range")

        for item in status_items:
            print(f"  {item}")

        print(f"\n{'='*80}")

        # Save results to JSON
        self._save_results_to_file()

    def _save_results_to_file(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_results_{timestamp}.json"

        # Prepare data for JSON (convert numpy arrays)
        results = {
            'config': {
                'num_episodes': self.num_episodes,
                'steps_per_episode': self.steps_per_episode,
                'total_steps': self.total_steps,
                'soc_penalty_weight': self.cfg['soc_penalty_weight'],
                'bonus_constant': self.cfg['bonus_constant'],
                'max_step': self.cfg['max_step'],
                'flexibility_bonus_weight': self.cfg['flexibility_bonus_weight'],
                'soc_clipping_penalty': self.cfg['soc_clipping_penalty'],
            },
            'summary': {
                'improvement_rate': 100 * len(self.metrics['congestion']['improvements']) / self.total_steps,
                'worsening_rate': 100 * len(self.metrics['congestion']['worsenings']) / self.total_steps,
                'violations_per_step': (self.metrics['soc']['lower_violations'] +
                                      self.metrics['soc']['upper_violations']) / self.total_steps,
                'mean_soc': float(np.mean(self.metrics['soc']['history'])),
                'flex_band_percentage': 100 * np.sum(self.metrics['soc']['in_flex_band']) / (self.total_steps * 5),
                'avg_total_reward': float(np.mean(self.metrics['rewards']['total'])),
                'charge_percentage': 100 * self.metrics['actions']['charge_count'] /
                                   (self.metrics['actions']['charge_count'] +
                                    self.metrics['actions']['discharge_count'] +
                                    self.metrics['actions']['idle_count']),
            },
            'timestamp': timestamp,
            'duration_seconds': self.metrics['timing']['duration_seconds'],
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to: {filename}")


def main():
    """Main entry point for diagnostic test."""
    import warnings
    warnings.filterwarnings('ignore')

    print("\n" + "="*80)
    print("INTENSIVE DIAGNOSTIC TEST")
    print("="*80)
    print("\nThis test will run 15 episodes with 50 steps each (750 total steps)")
    print("to comprehensively evaluate the current configuration.")
    print("\nEstimated time: 4-6 minutes")
    print("\nPress Ctrl+C to interrupt if needed.")
    print("="*80 + "\n")

    try:
        # Run intensive test (extended to 15 episodes)
        framework = DiagnosticTestFramework(num_episodes=15, steps_per_episode=50)
        framework.run_comprehensive_test()

        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)

        sys.exit(0)

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
