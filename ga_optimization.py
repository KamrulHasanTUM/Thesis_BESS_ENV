"""
ga_optimization.py

Complete Genetic Algorithm system for optimizing BESS placement.
Includes: GA logic, visualization, formatting, and reporting.
"""

import numpy as np
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from ENV_BESS_main import ENV_BESS
from config import create_bess_env_config, load_config


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_ga_config():
    """Configuration for genetic algorithm."""
    return {
        # ========== GA Parameters ==========
        'population_size': 20,          # Number of placement configurations to test
        'num_generations': 10,          # How many evolution cycles
        'mutation_rate': 0.2,           # Probability of random bus change
        'crossover_rate': 0.7,          # Probability of combining parent placements
        'elite_size': 2,                # Keep top N placements unchanged
        
        # ========== RL Training per Placement ==========
        'ga_training_timesteps': 50_000,  # Quick training to evaluate each placement
        'num_bess': 5,                     # Number of BESS units
        
        # ========== Fitness Evaluation ==========
        'fitness_metric': 'congestion_reduction',
        'num_evaluation_episodes': 10,
    }


# ============================================================================
# GENETIC ALGORITHM CORE
# ============================================================================

class BESSPlacementGA:
    """Genetic Algorithm for optimizing BESS placement in power grid."""
    
    def __init__(self, env_config, ga_config):
        """Initialize GA with configuration."""
        self.env_config = env_config
        self.ga_config = ga_config
        self.num_bess = ga_config['num_bess']
        
        # Get valid bus indices (110kV buses)
        temp_env = ENV_BESS(**env_config)
        self.valid_buses = temp_env.net.bus[temp_env.net.bus['vn_kv'] == 110].index.values
        self.net = temp_env.net  # Keep reference for bus info
        print(f"GA initialized with {len(self.valid_buses)} valid 110kV buses")
        del temp_env
        
        # GA state
        self.population = []
        self.fitness_scores = []
        self.best_placement = None
        self.best_fitness = -np.inf
        self.history = []
    
    def create_random_placement(self):
        """Create random BESS placement (chromosome)."""
        return np.random.choice(self.valid_buses, size=self.num_bess, replace=False)
    
    def initialize_population(self):
        """Create initial population of random placements."""
        print("Initializing population...")
        self.population = [
            self.create_random_placement() 
            for _ in range(self.ga_config['population_size'])
        ]
    
    def evaluate_placement(self, placement):
        """
        Evaluate fitness of a BESS placement.
        Fitness = Average congestion reduction over evaluation episodes
        """
        env_config_copy = deepcopy(self.env_config)
        env = ENV_BESS(**env_config_copy)
        env.bess_locations = placement.astype(np.int32)
        
        congestion_scores = []
        
        for episode in range(self.ga_config['num_evaluation_episodes']):
            obs, info = env.reset()
            episode_loadings = []
            
            for step in range(50):
                action = np.random.uniform(-env.bess_power_mw, env.bess_power_mw, size=self.num_bess)
                obs, reward, terminated, truncated, info = env.step(action)
                max_loading = env.net.res_line['loading_percent'].max()
                episode_loadings.append(max_loading)
                
                if terminated or truncated:
                    break
            
            avg_loading = np.mean(episode_loadings)
            congestion_scores.append(100 - avg_loading)
        
        fitness = np.mean(congestion_scores)
        del env
        return fitness
    
    def evaluate_population(self):
        """Evaluate fitness of all placements in population."""
        print(f"Evaluating population of {len(self.population)}...")
        self.fitness_scores = []
        
        for i, placement in enumerate(self.population):
            fitness = self.evaluate_placement(placement)
            self.fitness_scores.append(fitness)
            print(f"  Placement {i+1}/{len(self.population)}: Fitness = {fitness:.2f}")
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_placement = placement.copy()
                print(f"  *** New best placement! Buses: {placement}, Fitness: {fitness:.2f}")
        
        self.fitness_scores = np.array(self.fitness_scores)
    
    def selection(self):
        """Select parents using tournament selection."""
        tournament_size = 3
        idx1 = np.random.choice(len(self.population), size=tournament_size, replace=False)
        idx2 = np.random.choice(len(self.population), size=tournament_size, replace=False)
        
        parent1 = self.population[idx1[np.argmax(self.fitness_scores[idx1])]]
        parent2 = self.population[idx2[np.argmax(self.fitness_scores[idx2])]]
        
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        """Combine two parent placements to create offspring."""
        if random.random() < self.ga_config['crossover_rate']:
            point = random.randint(1, self.num_bess - 1)
            child = np.concatenate([parent1[:point], parent2[point:]])
            
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
        """Randomly change one bus in placement."""
        if random.random() < self.ga_config['mutation_rate']:
            idx = random.randint(0, self.num_bess - 1)
            available = [b for b in self.valid_buses if b not in placement]
            placement[idx] = random.choice(available)
        return placement
    
    def evolve(self):
        """Create next generation through selection, crossover, mutation."""
        elite_indices = np.argsort(self.fitness_scores)[-self.ga_config['elite_size']:]
        new_population = [self.population[i].copy() for i in elite_indices]
        
        while len(new_population) < self.ga_config['population_size']:
            parent1, parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
    
    def run(self):
        """Run the genetic algorithm."""
        print("="*70)
        print("Starting Genetic Algorithm for BESS Placement Optimization")
        print("="*70)
        
        self.initialize_population()
        
        for generation in range(self.ga_config['num_generations']):
            print(f"\n{'='*70}")
            print(f"Generation {generation + 1}/{self.ga_config['num_generations']}")
            print(f"{'='*70}")
            
            self.evaluate_population()
            
            gen_stats = {
                'generation': generation + 1,
                'best_fitness': float(self.best_fitness),
                'avg_fitness': float(np.mean(self.fitness_scores)),
                'best_placement': self.best_placement.tolist()
            }
            self.history.append(gen_stats)
            
            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Best Fitness: {self.best_fitness:.2f}")
            print(f"  Avg Fitness: {np.mean(self.fitness_scores):.2f}")
            print(f"  Best Placement: {self.best_placement}")
            
            if generation < self.ga_config['num_generations'] - 1:
                self.evolve()
        
        self.save_results()
        self.create_visualizations()
        self.format_results()
        
        print(f"\n{'='*70}")
        print("GA Optimization Complete!")
        print(f"{'='*70}")
        
        return self.best_placement
    
    def save_results(self):
        """Save GA results to JSON."""
        results = {
            'best_placement': self.best_placement.tolist(),
            'best_fitness': float(self.best_fitness),
            'history': self.history,
            'ga_config': self.ga_config
        }
        
        with open('ga_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\n✓ Results saved to ga_results.json")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def create_visualizations(self):
        """Generate all visualization charts."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        try:
            self._plot_fitness_evolution()
            self._plot_improvement_analysis()
            self._plot_placement_comparison()
            self._plot_convergence()
            print("\n✓ All visualizations created successfully")
        except Exception as e:
            print(f"\n✗ Visualization error: {e}")
    
    def _plot_fitness_evolution(self):
        """Plot fitness improvement over generations."""
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-o', linewidth=2, markersize=8, label='Best Fitness')
        plt.plot(generations, avg_fitness, 'r--s', linewidth=2, markersize=6, label='Average Fitness')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Score', fontsize=12)
        plt.title('GA Optimization Progress: BESS Placement', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ga_fitness_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ ga_fitness_evolution.png")
    
    def _plot_improvement_analysis(self):
        """Plot absolute and percentage improvements."""
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        initial = best_fitness[0]
        improvements = [(f - initial) for f in best_fitness]
        improvement_pct = [(f - initial) / abs(initial) * 100 for f in best_fitness]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(generations, improvements, color='green', alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Fitness Improvement (Absolute)', fontsize=11)
        ax1.set_title('Absolute Fitness Gain', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.bar(generations, improvement_pct, color='blue', alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Fitness Improvement (%)', fontsize=11)
        ax2.set_title('Percentage Fitness Gain', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('ga_improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ ga_improvement_analysis.png")
    
    def _plot_placement_comparison(self):
        """Compare initial vs optimized placement."""
        initial = self.history[0]['best_placement']
        optimized = self.best_placement
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(range(len(initial)), initial, color='red', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('BESS Unit', fontsize=11)
        ax1.set_ylabel('Bus Index', fontsize=11)
        ax1.set_title(f'Initial Placement\nFitness: {self.history[0]["best_fitness"]:.2f}', 
                      fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(initial)))
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.bar(range(len(optimized)), optimized, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('BESS Unit', fontsize=11)
        ax2.set_ylabel('Bus Index', fontsize=11)
        ax2.set_title(f'GA-Optimized Placement\nFitness: {self.best_fitness:.2f}', 
                      fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(optimized)))
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('ga_placement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ ga_placement_comparison.png")
    
    def _plot_convergence(self):
        """Analyze convergence rate."""
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        changes = [0] + [best_fitness[i] - best_fitness[i-1] for i in range(1, len(best_fitness))]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(generations, best_fitness, 'b-o', linewidth=2, markersize=8)
        ax1.set_ylabel('Best Fitness', fontsize=11)
        ax1.set_title('GA Convergence Analysis', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(generations, changes, color='purple', alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Fitness Change (Δ)', fontsize=11)
        ax2.set_title('Rate of Improvement', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('ga_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ ga_convergence_analysis.png")
    
    # ========================================================================
    # RESULTS FORMATTING
    # ========================================================================
    
    def format_results(self):
        """Create formatted text report and CSV."""
        print("\n" + "="*70)
        print("FORMATTING RESULTS")
        print("="*70)
        
        bus_details = self._get_bus_details()
        self._create_text_report(bus_details)
        self._create_csv_table(bus_details)
    
    def _get_bus_details(self):
        """Extract detailed bus information."""
        details = []
        for i, bus_idx in enumerate(self.best_placement, 1):
            bus_row = self.net.bus.loc[bus_idx]
            details.append({
                'bess_num': i,
                'bus_index': int(bus_idx),
                'bus_name': bus_row.get('name', f'Bus_{bus_idx}'),
                'voltage_kv': float(bus_row.get('vn_kv', 110.0))
            })
        return details
    
    def _create_text_report(self, bus_details):
        """Create formatted text report."""
        lines = []
        lines.append("="*70)
        lines.append("SOLUTION ANALYSIS")
        lines.append("="*70)
        lines.append("")
        lines.append("Optimal BESS Placement:")
        
        for bus in bus_details:
            bus_info = f" ({bus['bus_name']}, {bus['voltage_kv']:.1f}kV)"
            lines.append(f"  BESS {bus['bess_num']}: Bus {bus['bus_index']}{bus_info}")
        
        lines.append("")
        lines.append(f"Fitness Score: {self.best_fitness:.2f}")
        lines.append("")
        lines.append("="*70)
        lines.append("")
        lines.append("Configuration Format (for config.py):")
        lines.append(f"'bess_locations': {self.best_placement.tolist()},")
        lines.append("")
        
        # Add generation summary
        lines.append("="*70)
        lines.append("EVOLUTION SUMMARY")
        lines.append("="*70)
        for h in self.history:
            lines.append(f"Gen {h['generation']:2d}: Best={h['best_fitness']:6.2f}, Avg={h['avg_fitness']:6.2f}")
        lines.append("="*70)
        
        report = "\n".join(lines)
        print(report)
        
        with open('ga_optimization_report.txt', 'w') as f:
            f.write(report)
        print("\n✓ ga_optimization_report.txt")
    
    def _create_csv_table(self, bus_details):
        """Create CSV table of placement."""
        df = pd.DataFrame(bus_details)
        df.columns = ['BESS Unit', 'Bus Index', 'Bus Name', 'Voltage (kV)']
        df.to_csv('optimal_bess_placement.csv', index=False)
        print("✓ optimal_bess_placement.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run GA optimization."""
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)
    ga_config = get_ga_config()
    
    ga = BESSPlacementGA(env_config, ga_config)
    optimal_placement = ga.run()
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOptimal BESS placement: {optimal_placement}")
    print("\nFiles generated:")
    print("  • ga_results.json")
    print("  • ga_fitness_evolution.png")
    print("  • ga_improvement_analysis.png")
    print("  • ga_placement_comparison.png")
    print("  • ga_convergence_analysis.png")
    print("  • ga_optimization_report.txt")
    print("  • optimal_bess_placement.csv")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()