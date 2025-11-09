"""
ga_optimization_improved.py

Improved GA optimization with better fitness function.
Uses RANDOM POLICY baseline to find placements where BESS physically helps.

Key improvements:
1. Fitness = Success rate of random actions (should be > 55%)
2. Tests physical impact, not trained agent performance
3. Faster evaluation (no RL training needed)
"""

import numpy as np
import random
import json
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config, load_config


def get_improved_ga_config():
    """Improved GA configuration."""
    return {
        # ========== GA Parameters ==========
        'population_size': 30,          # Increased from 20 for better diversity
        'num_generations': 15,          # Increased from 10 for better convergence
        'mutation_rate': 0.25,          # Slightly higher for more exploration
        'crossover_rate': 0.7,
        'elite_size': 3,                # Keep top 3 placements

        # ========== Fitness Evaluation ==========
        'num_bess': 5,
        'num_evaluation_episodes': 20,   # Test with 20 random episodes
        'steps_per_episode': 50,

        # ========== BESS Capacity ==========
        'bess_capacity_mwh': 50.0,      # Keep at 50 MW (economic constraint)
        'bess_power_mw': 50.0,
    }


class ImprovedBESSPlacementGA:
    """Improved GA using random policy baseline for fitness."""

    def __init__(self, env_config, ga_config):
        """Initialize GA."""
        self.env_config = env_config
        self.ga_config = ga_config
        self.num_bess = ga_config['num_bess']

        # Override BESS capacity in env_config
        self.env_config['bess_capacity_mwh'] = ga_config['bess_capacity_mwh']
        self.env_config['bess_power_mw'] = ga_config['bess_power_mw']

        # Get valid buses
        temp_env = ENV_BESS(**env_config)
        self.valid_buses = temp_env.net.bus[temp_env.net.bus['vn_kv'] == 110].index.values
        print(f"GA initialized with {len(self.valid_buses)} valid 110kV buses")
        print(f"BESS capacity: {ga_config['bess_capacity_mwh']} MWh, {ga_config['bess_power_mw']} MW")
        del temp_env

        # GA state
        self.population = []
        self.fitness_scores = []
        self.best_placement = None
        self.best_fitness = -np.inf
        self.history = []

    def create_random_placement(self):
        """Create random BESS placement."""
        return np.random.choice(self.valid_buses, size=self.num_bess, replace=False)

    def initialize_population(self):
        """Create initial population."""
        print("Initializing population...")
        self.population = [
            self.create_random_placement()
            for _ in range(self.ga_config['population_size'])
        ]

    def evaluate_placement_random_baseline(self, placement):
        """
        Evaluate placement using RANDOM POLICY BASELINE.

        Fitness = Success rate of random actions
        - Success = action reduces max loading
        - Target: > 55% success rate for good placement
        """
        from copy import deepcopy

        env_config_copy = deepcopy(self.env_config)
        env_config_copy['bess_locations'] = placement.tolist()
        env = ENV_BESS(**env_config_copy)

        successes = []
        deltas = []

        for episode in range(self.ga_config['num_evaluation_episodes']):
            obs, info = env.reset()

            for step in range(self.ga_config['steps_per_episode']):
                # Random action in normalized range [-1, +1]
                action = np.random.uniform(-1.0, 1.0, size=self.num_bess)

                obs, reward, terminated, truncated, info = env.step(action)

                # Measure impact
                try:
                    loading_before = env.loading_before_action
                    loading_after = env.loading_after_action
                    delta = loading_before - loading_after

                    deltas.append(delta)
                    if delta > 0:
                        successes.append(1)
                    else:
                        successes.append(0)
                except:
                    pass

                if terminated or truncated:
                    break

        # Fitness = success rate (percentage of actions that help)
        if len(successes) > 0:
            success_rate = sum(successes) / len(successes)
            mean_delta = np.mean(deltas)

            # Fitness combines success rate and mean improvement
            fitness = success_rate * 100 + mean_delta  # Success rate dominant
        else:
            fitness = 0.0
            success_rate = 0.0
            mean_delta = 0.0

        del env

        return fitness, success_rate, mean_delta

    def evaluate_population(self):
        """Evaluate all placements."""
        print(f"\nEvaluating population of {len(self.population)}...")
        self.fitness_scores = []

        for i, placement in enumerate(self.population):
            fitness, success_rate, mean_delta = self.evaluate_placement_random_baseline(placement)
            self.fitness_scores.append(fitness)

            print(f"  [{i+1}/{len(self.population)}] Buses: {placement}")
            print(f"      Success Rate: {success_rate:.1%}, Mean Delta: {mean_delta:+.3f}%, Fitness: {fitness:.2f}")

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_placement = placement.copy()
                self.best_success_rate = success_rate
                self.best_delta = mean_delta
                print(f"      *** NEW BEST! ***")

        self.fitness_scores = np.array(self.fitness_scores)

    def selection(self):
        """Tournament selection."""
        tournament_size = 3
        idx1 = np.random.choice(len(self.population), size=tournament_size, replace=False)
        idx2 = np.random.choice(len(self.population), size=tournament_size, replace=False)

        parent1 = self.population[idx1[np.argmax(self.fitness_scores[idx1])]]
        parent2 = self.population[idx2[np.argmax(self.fitness_scores[idx2])]]

        return parent1, parent2

    def crossover(self, parent1, parent2):
        """Crossover operation."""
        if random.random() < self.ga_config['crossover_rate']:
            point = random.randint(1, self.num_bess - 1)
            child = np.concatenate([parent1[:point], parent2[point:]])

            # Remove duplicates
            child = np.unique(child)
            if len(child) < self.num_bess:
                available = [b for b in self.valid_buses if b not in child]
                needed = self.num_bess - len(child)
                child = np.concatenate([child, np.random.choice(available, needed, replace=False)])
            elif len(child) > self.num_bess:
                child = child[:self.num_bess]

            return child.astype(np.int32)
        else:
            return parent1.copy()

    def mutate(self, placement):
        """Mutation operation."""
        if random.random() < self.ga_config['mutation_rate']:
            idx = random.randint(0, self.num_bess - 1)
            available = [b for b in self.valid_buses if b not in placement]
            placement[idx] = random.choice(available)
        return placement

    def evolve_generation(self):
        """Create next generation."""
        # Elitism: keep best placements
        elite_indices = np.argsort(self.fitness_scores)[-self.ga_config['elite_size']:]
        new_population = [self.population[i].copy() for i in elite_indices]

        # Generate rest through selection, crossover, mutation
        while len(new_population) < self.ga_config['population_size']:
            parent1, parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population

    def run(self):
        """Run GA optimization."""
        print("="*80)
        print(" IMPROVED GA OPTIMIZATION - Random Baseline Fitness ".center(80, "="))
        print("="*80)

        self.initialize_population()

        for generation in range(self.ga_config['num_generations']):
            print(f"\n{'='*80}")
            print(f" Generation {generation + 1}/{self.ga_config['num_generations']} ".center(80, "="))
            print(f"{'='*80}")

            self.evaluate_population()

            # Record history
            self.history.append({
                'generation': generation + 1,
                'best_fitness': float(self.best_fitness),
                'best_placement': self.best_placement.tolist(),
                'best_success_rate': float(self.best_success_rate),
                'best_delta': float(self.best_delta),
                'mean_fitness': float(np.mean(self.fitness_scores)),
                'std_fitness': float(np.std(self.fitness_scores))
            })

            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Best Fitness: {self.best_fitness:.2f}")
            print(f"  Best Success Rate: {self.best_success_rate:.1%}")
            print(f"  Best Placement: {self.best_placement}")
            print(f"  Population Mean Fitness: {np.mean(self.fitness_scores):.2f}")

            # Evolve to next generation
            if generation < self.ga_config['num_generations'] - 1:
                self.evolve_generation()

        return self.best_placement, self.best_fitness, self.best_success_rate


def run_improved_ga_optimization():
    """Main function to run improved GA."""

    # Load config
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)
    ga_config = get_improved_ga_config()

    # Run GA
    ga = ImprovedBESSPlacementGA(env_config, ga_config)
    best_placement, best_fitness, best_success_rate = ga.run()

    # Final report
    print("\n" + "="*80)
    print(" OPTIMIZATION COMPLETE ".center(80, "="))
    print("="*80)
    print(f"\nBest BESS Placement: {best_placement}")
    print(f"Best Success Rate: {best_success_rate:.1%}")
    print(f"Best Fitness: {best_fitness:.2f}")

    # Save results
    results = {
        'best_placement': best_placement.tolist(),
        'best_success_rate': float(best_success_rate),
        'best_fitness': float(best_fitness),
        'bess_capacity_mwh': ga_config['bess_capacity_mwh'],
        'bess_power_mw': ga_config['bess_power_mw'],
        'history': ga.history
    }

    with open('ga_results_improved.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: ga_results_improved.json")

    # Evaluation
    print("\n" + "="*80)
    print(" EVALUATION ".center(80, "="))
    print("="*80)

    if best_success_rate > 0.55:
        print("[PASS] Success rate > 55% - Good placement for RL training!")
        print("Action: Update config.py with this placement and train agent.")
    elif best_success_rate > 0.48:
        print("[MARGINAL] Success rate 48-55% - May need larger BESS capacity.")
        print("Action: Consider increasing to 100 MW or re-running GA.")
    else:
        print("[FAIL] Success rate < 48% - BESS cannot help at these locations.")
        print("Action: Increase BESS capacity to 100 MW and re-run GA.")

    print("\n" + "="*80)

    return results


if __name__ == "__main__":
    results = run_improved_ga_optimization()
