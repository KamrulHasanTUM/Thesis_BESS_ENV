"""
wandb_integration.py

Weights & Biases integration for ENV_BESS training tracking.
Logs hyperparameters, episode metrics, BESS state, and grid performance.
"""

import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback(BaseCallback):
    """Custom callback for logging ENV_BESS metrics to W&B."""
    
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        """Called at every environment step."""
        if self.n_calls % self.log_freq == 0:
            # Access the actual environment (handle vectorized envs)
            try:
                if hasattr(self.training_env, 'envs'):
                    env = self.training_env.envs[0].unwrapped
                else:
                    env = self.training_env.unwrapped
                
                # Log grid metrics
                if hasattr(env, 'net') and hasattr(env.net, 'res_line'):
                    wandb.log({
                        'step': self.num_timesteps,
                        'grid/max_loading': float(env.net.res_line['loading_percent'].max()),
                        'grid/avg_loading': float(env.net.res_line['loading_percent'].mean()),
                    }, step=self.num_timesteps)
                
                # Log BESS state
                if hasattr(env, 'bess_soc'):
                    wandb.log({
                        'bess/avg_soc': float(np.mean(env.bess_soc)),
                        'bess/avg_power': float(np.mean(np.abs(env.bess_power))),
                    }, step=self.num_timesteps)
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"W&B logging error: {e}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (after n_steps)."""
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            
            wandb.log({
                'rollout/ep_reward_mean': ep_info.get('r', 0),
                'rollout/ep_len_mean': ep_info.get('l', 0),
            }, step=self.num_timesteps)


def init_wandb_run(config, project_name="bess-rl-training"):
    """
    Initialize W&B run with experiment configuration.
    
    Args:
        config: Dictionary containing all hyperparameters
        project_name: W&B project name
        
    Returns:
        wandb.run object
    """
    run = wandb.init(
        project=project_name,
        name=f"{config['exp_name']}_{config['exp_id']}",
        config=config,
        sync_tensorboard=False,  # Set True if using TensorBoard too
        monitor_gym=True,
        save_code=True,
    )
    
    # Log environment configuration
    wandb.config.update({
        "environment": "ENV_BESS",
        "algorithm": "PPO",
        "network": config.get('simbench_code', 'unknown'),
    })
    
    return run


def log_episode_summary(episode_data):
    """
    Log detailed episode summary to W&B.
    
    Args:
        episode_data: Dict with episode metrics
    """
    wandb.log({
        'episode/total_reward': episode_data['total_reward'],
        'episode/length': episode_data['length'],
        'episode/final_max_loading': episode_data['final_max_loading'],
        'episode/avg_soc': episode_data['avg_soc'],
        'episode/energy_throughput': episode_data['energy_throughput'],
    })


def log_bess_heatmap(soc_history, step):
    """
    Log BESS SoC evolution as heatmap.
    
    Args:
        soc_history: Array of shape (timesteps, num_bess)
        step: Current training step
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(soc_history.T, cmap='RdYlGn', vmin=0, vmax=1, 
                cbar_kws={'label': 'State of Charge'}, ax=ax)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('BESS Unit')
    ax.set_title('BESS SoC Evolution Over Episode')
    
    wandb.log({"bess/soc_heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_grid_topology(net, step):
    """
    Log grid network topology visualization.
    
    Args:
        net: Pandapower network object
        step: Current training step
    """
    try:
        import pandapower.plotting as plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot.simple_plot(net, ax=ax, plot_loads=True, plot_sgens=True)
        ax.set_title('Grid Network Topology')
        
        wandb.log({"grid/topology": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        print(f"Could not log grid topology: {e}")