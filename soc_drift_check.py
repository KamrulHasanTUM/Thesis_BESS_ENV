"""Utility to run short BESS rollouts and inspect SoC and reward telemetry."""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from config import create_bess_env_config, load_config
from ENV_BESS_main import ENV_BESS


def parse_action(action_values: Sequence[float] | None, num_bess: int) -> np.ndarray:
    """Return an action vector of length ``num_bess`` normalized [-1, +1].

    The user can pass a single value to broadcast across all units, or ``num_bess``
    comma-separated values to target individual batteries.
    """
    if action_values is None:
        return np.zeros(num_bess, dtype=np.float32)

    values = np.asarray(action_values, dtype=np.float32)
    if values.size == 1:
        return np.repeat(values, num_bess)

    if values.size != num_bess:
        raise ValueError(
            f"Expected 1 or {num_bess} action values, received {values.size}."
        )
    return values


def run_sequence(env: ENV_BESS, action: np.ndarray, warmup: int, steps: int) -> None:
    """Execute a rollout that first applies zero power, then a constant action."""
    obs, info = env.reset()
    print("Environment reset. Observation shape:", np.shape(obs))

    zero_action = np.zeros(env.num_bess, dtype=np.float32)

    for step in range(warmup):
        _, reward, terminated, truncated, info = env.step(zero_action.tolist())
        print(
            f"Warmup step {step + 1}: SoC={info['bess_soc']}, "
            f"reward={reward:.4f}, parts={info['reward_breakdown']}"
        )
        if terminated or truncated:
            print("Episode ended during warmup.")
            return

    for step in range(steps):
        _, reward, terminated, truncated, info = env.step(action.tolist())
        print(
            f"Action step {step + 1}: SoC={info['bess_soc']}, "
            f"reward={reward:.4f}, parts={info['reward_breakdown']}"
        )
        if terminated or truncated:
            print("Episode ended early during action phase.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect SoC drift and reward components under a fixed BESS dispatch.",
    )
    parser.add_argument(
        "--action",
        nargs="*",
        type=float,
        help=(
            "Action values in MW. Provide one value to broadcast to all BESS or "
            "one value per unit. Defaults to zero power."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of zero-action steps to run before applying the test action.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Number of steps to apply the specified action after the warmup phase.",
    )
    args = parser.parse_args()

    init_meta = load_config()
    env_config = create_bess_env_config(init_meta)
    env = ENV_BESS(**env_config)

    action = parse_action(args.action, env.num_bess)
    print(
        "Running rollout with action vector:",
        action,
        "(MW per unit) for",
        args.steps,
        "steps after",
        args.warmup,
        "warmup steps",
    )

    run_sequence(env, action, warmup=args.warmup, steps=args.steps)


if __name__ == "__main__":
    main()