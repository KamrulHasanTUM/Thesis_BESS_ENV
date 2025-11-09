"""
run_ga_with_trained_agent.py

Run GA optimization using your trained PPO agent instead of random actions.
This finds optimal BESS locations for your trained policy.
"""

from ga_optimization import BESSPlacementGA, get_ga_config
from config import load_config, create_bess_env_config

def main():
    """Run GA with trained agent."""
    print("="*70)
    print("GA OPTIMIZATION WITH TRAINED AGENT")
    print("="*70)
    
    # Load configurations
    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)
    ga_config = get_ga_config()
    
    # Path to your trained model
    trained_model_path = "final_model.zip"  # ← Change if your model has different name
    
    print(f"\nUsing trained model: {trained_model_path}")
    print(f"Population size: {ga_config['population_size']}")
    print(f"Generations: {ga_config['num_generations']}")
    print(f"Evaluation episodes per placement: {ga_config['num_evaluation_episodes']}")
    print("\n" + "="*70 + "\n")
    
    # Create GA instance
    ga = BESSPlacementGA(env_config, ga_config)
    
    # CRITICAL: Override evaluate_placement to use trained model
    original_evaluate = ga.evaluate_placement
    ga.evaluate_placement = lambda p: original_evaluate(p, model_path=trained_model_path)
    
    # Run optimization
    optimal_placement = ga.run()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Optimal BESS placement: {optimal_placement}")
    print(f"Fitness score: {ga.best_fitness:.2f}")
    print("\nℹ️  Copy this to config.py:")
    print(f"'bess_locations': {optimal_placement.tolist()},")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()