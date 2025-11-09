"""
Advanced Genetic Algorithm for BESS Placement Optimization - FIXED VERSION
"""

import numpy as np
import pandas as pd
import simbench as sb
import pandapower as pp
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')


class BESSPlacementOptimizer:
    """Advanced GA optimizer for BESS placement."""
    
    def __init__(self, 
                 simbench_code="1-HV-mixed--0-sw",
                 case_study='hL',
                 num_bess=5,
                 bess_power_mw=50.0,
                 bess_capacity_mwh=50.0,
                 population_size=80,
                 num_generations=40,
                 num_test_timesteps=15,
                 verbose=True):
        
        self.simbench_code = simbench_code
        self.case_study = case_study
        self.num_bess = num_bess
        self.bess_power_mw = bess_power_mw
        self.bess_capacity_mwh = bess_capacity_mwh
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_test_timesteps = num_test_timesteps
        self.verbose = verbose
        
        # Load network with study case
        if self.verbose:
            print(f"Loading network: {simbench_code}, case: {case_study}")
        
        self.net = sb.get_simbench_net(simbench_code)
        
        # Load profiles
        self.load_data, self.sgen_data = self._load_case_data()
        
        # Get candidate bus locations
        self.candidate_buses = self._get_candidate_buses()
        
        if self.verbose:
            print(f"Network: {simbench_code}")
            print(f"Case study: {case_study}")
            print(f"Total buses: {len(self.net.bus)}")
            print(f"Candidate HV buses: {len(self.candidate_buses)}")
            print(f"Load data shape: {self.load_data.shape}")
            print(f"Sgen data shape: {self.sgen_data.shape}")
            print(f"BESS to place: {num_bess}")
            print(f"Population size: {population_size}")
            print(f"Generations: {num_generations}")
    
    def _load_case_data(self):
        """Load time-series data for the case study - FINAL FIX."""
    
        # Get profiles
        profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=False)
    
        if self.verbose:
            print(f"Available profile keys: {list(profiles.keys())[:10]}")  # Show first 10
    
        # Profiles structure: MultiIndex with tuples like ('load', 'p_mw')
        # We need to extract the DataFrame for loads and sgens
    
        # Method 1: Try direct access
        try:
        # Get columns that start with element type
         load_cols = [col for col in profiles.columns if col[0] == 'load']
         sgen_cols = [col for col in profiles.columns if col[0] == 'sgen']
        
         if load_cols and sgen_cols:
            # Extract just the p_mw columns
            load_profile = profiles[load_cols]
            sgen_profile = profiles[sgen_cols]
            
            # Simplify column names to just indices
            load_profile.columns = range(len(load_cols))
            sgen_profile.columns = range(len(sgen_cols))
            
            if self.verbose:
                print(f"Loaded profiles: Load shape={load_profile.shape}, Sgen shape={sgen_profile.shape}")
            
            return load_profile, sgen_profile
        except Exception as e:
         if self.verbose:
            print(f"Method 1 failed: {e}")
    
        # Method 2: Build from scratch using helper function
        try:
         from env_helpers import load_simbench_profiles_and_cases, prepare_train_test_split
        
         net_copy, profiles_dict = load_simbench_profiles_and_cases(self.net, self.case_study)
         data_splits = prepare_train_test_split(profiles_dict)
        
         load_profile = pd.concat([data_splits['load_train'], data_splits['load_test']])
         sgen_profile = pd.concat([data_splits['sgen_train'], data_splits['sgen_test']])
        
         if self.verbose:
            print(f"Loaded using env_helpers: Load shape={load_profile.shape}, Sgen shape={sgen_profile.shape}")
        
         return load_profile, sgen_profile
        
        except Exception as e:
         if self.verbose:
            print(f"Method 2 failed: {e}")
        raise ValueError("Cannot load profiles. Try checking SimBench version or network data.")
    
    def _get_candidate_buses(self):
        """Get candidate bus locations (high voltage only)."""
        # High voltage buses (>= 110 kV)
        hv_buses = self.net.bus[self.net.bus['vn_kv'] >= 110].index.tolist()
        
        # Exclude slack buses
        slack_buses = self.net.ext_grid['bus'].tolist()
        valid_buses = [b for b in hv_buses if b not in slack_buses]
        
        return sorted(valid_buses)
    
    def _apply_timestep(self, net, timestep_idx):
        """Apply load and generation data for a specific timestep."""
        net_copy = net.deepcopy()
        
        # Apply load data
        for load_idx in net_copy.load.index:
            if load_idx < self.load_data.shape[1]:  # Check column exists
                net_copy.load.at[load_idx, 'p_mw'] = self.load_data.iloc[timestep_idx, load_idx]
        
        # Apply generator data
        for sgen_idx in net_copy.sgen.index:
            if sgen_idx < self.sgen_data.shape[1]:  # Check column exists
                net_copy.sgen.at[sgen_idx, 'p_mw'] = self.sgen_data.iloc[timestep_idx, sgen_idx]
        
        return net_copy
    
    def _simulate_bess_action(self, net, bess_locations, action_strategy='adaptive'):
        """Simulate BESS dispatch and return loading reduction."""
        
        # Get baseline loading
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, max_iteration=100)
            loading_before = net.res_line['loading_percent'].max()
        except:
            return -999.0
        
        # Add BESS
        net_with_bess = net.deepcopy()
        
        for bus in bess_locations:
            if action_strategy == 'discharge':
                power = self.bess_power_mw
            elif action_strategy == 'charge':
                power = -self.bess_power_mw
            elif action_strategy == 'adaptive':
                if loading_before > 30:
                    power = self.bess_power_mw * 0.8
                elif loading_before < 15:
                    power = -self.bess_power_mw * 0.5
                else:
                    power = self.bess_power_mw * 0.4
            else:  # balanced
                power = self.bess_power_mw * 0.5
            
            pp.create_sgen(net_with_bess, bus=bus, p_mw=power, q_mvar=0, 
                          name=f"BESS_{bus}", controllable=False)
        
        # Run power flow with BESS
        try:
            pp.runpp(net_with_bess, algorithm='nr', calculate_voltage_angles=True, max_iteration=100)
            loading_after = net_with_bess.res_line['loading_percent'].max()
        except:
            return -999.0
        
        loading_reduction = loading_before - loading_after
        
        return loading_reduction
    
    def evaluate_individual(self, individual, test_timesteps):
        """Evaluate fitness of a BESS placement."""
        
        bess_locations = [self.candidate_buses[gene] for gene in individual]
        
        # Check for duplicates
        if len(set(bess_locations)) < len(bess_locations):
            return (-999.0,)
        
        reductions = []
        
        for timestep_idx in test_timesteps:
            net_t = self._apply_timestep(self.net, timestep_idx)
            
            # Test multiple strategies
            for strategy in ['adaptive', 'discharge', 'balanced']:
                reduction = self._simulate_bess_action(net_t, bess_locations, strategy)
                if reduction > -900:  # Valid result
                    reductions.append(reduction)
        
        if len(reductions) == 0:
            return (-999.0,)
        
        # Fitness = average reduction + consistency bonus
        avg_reduction = np.mean(reductions)
        positive_ratio = np.sum([r > 0 for r in reductions]) / len(reductions)
        consistency_bonus = positive_ratio * 2.0
        
        fitness = avg_reduction + consistency_bonus
        
        return (fitness,)
    
    def run_optimization(self):
        """Run genetic algorithm optimization."""
        
        # Sample timesteps
        all_timesteps = list(range(min(1000, len(self.load_data))))  # Limit to 1000 timesteps
        test_timesteps = random.sample(all_timesteps, min(self.num_test_timesteps, len(all_timesteps)))
        
        # Setup DEAP
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        toolbox.register("attr_int", random.randint, 0, len(self.candidate_buses) - 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_int, n=self.num_bess)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, 
                        low=0, up=len(self.candidate_buses)-1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate_individual, test_timesteps=test_timesteps)
        
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        
        population = toolbox.population(n=self.population_size)
        
        if self.verbose:
            print("\nStarting GA optimization...")
            print("="*60)
        
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=0.7,
            mutpb=0.2,
            ngen=self.num_generations,
            stats=stats,
            verbose=self.verbose
        )
        
        best_individual = tools.selBest(population, k=1)[0]
        best_locations = [self.candidate_buses[gene] for gene in best_individual]
        best_fitness = best_individual.fitness.values[0]
        
        return best_locations, best_fitness, logbook
    
    def validate_solution(self, bess_locations, num_validation_timesteps=50):
        """Validate the optimized solution."""
        
        print("\n" + "="*60)
        print("VALIDATING OPTIMIZED BESS LOCATIONS")
        print("="*60)
        print(f"Locations: {bess_locations}")
        
        all_timesteps = list(range(min(1000, len(self.load_data))))
        validation_timesteps = random.sample(
            all_timesteps, 
            min(num_validation_timesteps, len(all_timesteps))
        )
        
        results = {
            'adaptive': [],
            'discharge': [],
            'charge': [],
            'balanced': []
        }
        
        for timestep_idx in validation_timesteps:
            net_t = self._apply_timestep(self.net, timestep_idx)
            
            for strategy in results.keys():
                reduction = self._simulate_bess_action(net_t, bess_locations, strategy)
                if reduction > -900:
                    results[strategy].append(reduction)
        
        # Print results
        print(f"\nValidation on {num_validation_timesteps} timesteps:")
        for strategy, reductions in results.items():
            if len(reductions) > 0:
                avg = np.mean(reductions)
                std = np.std(reductions)
                success_rate = np.sum([r > 0 for r in reductions]) / len(reductions) * 100
                
                print(f"\n  {strategy.capitalize()} Strategy:")
                print(f"    Avg Reduction: {avg:+.3f}% (±{std:.3f})")
                print(f"    Success Rate:  {success_rate:.1f}%")
                print(f"    Best:  {max(reductions):+.3f}%")
                print(f"    Worst: {min(reductions):+.3f}%")
        
        # Overall assessment
        all_adaptive = results['adaptive']
        if len(all_adaptive) > 0:
            overall_avg = np.mean(all_adaptive)
            overall_success = np.sum([r > 0 for r in all_adaptive]) / len(all_adaptive) * 100
        else:
            overall_avg = -999
            overall_success = 0
        
        print("\n" + "="*60)
        print("ASSESSMENT")
        print("="*60)
        
        if overall_avg > 1.0 and overall_success > 60:
            print("✅ EXCELLENT LOCATIONS")
            print(f"   Average reduction: {overall_avg:+.2f}%")
            print(f"   Success rate: {overall_success:.1f}%")
            print("   → Proceed with RL training")
            assessment = "excellent"
        elif overall_avg > 0.5 and overall_success > 55:
            print("✅ GOOD LOCATIONS")
            print(f"   Average reduction: {overall_avg:+.2f}%")
            print(f"   Success rate: {overall_success:.1f}%")
            print("   → Should work for RL training")
            assessment = "good"
        elif overall_avg > 0.2:
            print("⚠️  MARGINAL LOCATIONS")
            print(f"   Average reduction: {overall_avg:+.2f}%")
            print(f"   Success rate: {overall_success:.1f}%")
            print("   → Consider increasing BESS capacity or re-optimizing")
            assessment = "marginal"
        else:
            print("❌ POOR LOCATIONS")
            print(f"   Average reduction: {overall_avg:+.2f}%")
            print(f"   Success rate: {overall_success:.1f}%")
            print("   → Re-run optimization or change network parameters")
            assessment = "poor"
        
        return assessment, overall_avg, overall_success


def main():
    """Run BESS placement optimization."""
    
    print("="*60)
    print("ADVANCED BESS PLACEMENT OPTIMIZATION")
    print("="*60)
    
    optimizer = BESSPlacementOptimizer(
        simbench_code="1-HV-mixed--0-sw",
        case_study='hL',
        num_bess=5,
        bess_power_mw=50.0,
        bess_capacity_mwh=50.0,
        population_size=60,  # Reduced for faster testing
        num_generations=30,  # Reduced for faster testing
        num_test_timesteps=10,  # Reduced for faster testing
        verbose=True
    )
    
    best_locations, best_fitness, logbook = optimizer.run_optimization()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best locations: {best_locations}")
    print(f"Best fitness: {best_fitness:.3f}")
    
    assessment, avg_reduction, success_rate = optimizer.validate_solution(
        best_locations, 
        num_validation_timesteps=40
    )
    
    # Save results
    results = {
        'locations': best_locations,
        'fitness': float(best_fitness),
        'assessment': assessment,
        'avg_reduction': float(avg_reduction),
        'success_rate': float(success_rate)
    }
    
    import json
    with open('bess_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Results saved to 'bess_optimization_results.json'")
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if assessment in ['excellent', 'good']:
        print("1. Update config.py with new locations:")
        print(f"   'bess_locations': {best_locations}")
        print("2. Run training: python ENV_BESS_main.py")
    elif assessment == 'marginal':
        print("1. Try increasing BESS capacity:")
        print("   'bess_power_mw': 75.0 or 100.0")
        print("2. Or re-run GA with more generations")
    else:
        print("1. Re-run GA with different parameters")
        print("2. Or try more BESS units (num_bess=7)")
    
    return best_locations, best_fitness


if __name__ == "__main__":
    best_locations, fitness = main()