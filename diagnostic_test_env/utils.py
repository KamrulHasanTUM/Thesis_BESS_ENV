"""
utils.py

Utility functions and callbacks for reinforcement learning training.
Contains custom callbacks for logging, progress tracking, and learning rate scheduling.
"""

import time
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class TimeStepLoggingCallback(BaseCallback):
    """
    Custom callback for logging training metrics and error counts.
    Records start/end times, total timesteps, and environment-specific error counts.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.end_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.logger.record("start_time", self.start_time)

    def _on_training_end(self):
        self.end_time = time.time()
        self.logger.record("end_time", self.end_time)
        self.logger.record("total_time", self.end_time - self.start_time)
        if hasattr(self.training_env, "get_attr"):
            self.logger.record("total_load_flow_error_count", self.training_env.get_attr("convergence_error_count")[0])
            self.logger.record("total_line_disconnect_error_count", self.training_env.get_attr("line_disconnect_count")[0])
            self.logger.record("total_nan_vm_pu_error_count", self.training_env.get_attr("nan_vm_pu_count")[0])

    def _on_step(self):
        time_elapsed = time.time() - self.start_time
        self.logger.record("time_elapsed", time_elapsed)
        self.logger.record("total_timesteps", self.num_timesteps)
        # Ensure environment metrics are logged
        if hasattr(self.training_env, "get_attr"):
            self.logger.record("load_flow_error_count", self.training_env.get_attr("convergence_error_count")[0])
            self.logger.record("line_disconnect_error_count", self.training_env.get_attr("line_disconnect_count")[0])
            self.logger.record("nan_vm_pu_error_count", self.training_env.get_attr("nan_vm_pu_count")[0])

        return True


def linear_schedule(initial_value):
    """
    Returns a function that computes a linear decay of the learning rate.

    Args:
        initial_value: The initial learning rate value

    Returns:
        A function that takes progress_remaining (1.0 at start, 0.0 at end)
        and returns the linearly decayed learning rate
    """
    def func(progress_remaining):
        return progress_remaining * initial_value  # Linearly decrease LR
    return func


class TQDMProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress with TQDM progress bar.
    Displays real-time progress during model training and handles resumption correctly.
    """
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.last_timestep = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", leave=True)
        # If we're resuming training, update the progress bar to the current position
        if self.model.num_timesteps > 0:
            self.pbar.update(self.model.num_timesteps)
            self.last_timestep = self.model.num_timesteps

    def _on_step(self) -> bool:
        # Update progress bar by the difference since last update to avoid duplicate counting
        current_step = self.model.num_timesteps
        self.pbar.update(current_step - self.last_timestep)
        self.last_timestep = current_step
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
