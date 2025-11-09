#!/usr/bin/env python3
"""Utility to sample skill episode rewards under a flow module."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_flow_env_from_name


def load_module(module_path: str):
    spec = importlib.util.spec_from_file_location("reward_and_state", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def simulate(
    module_path: str,
    env_name: str,
    steps: int,
    seed: int,
) -> Dict[int, List[float]]:
    module_dict = load_module(module_path)
    env = make_craftax_flow_env_from_name(env_name, False, module_dict)
    params = env.default_params

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params)

    task_to_skill = np.array(env.task_to_skill_index)
    num_skills = int(task_to_skill.max()) + 1

    running = np.zeros((num_skills,), dtype=np.float32)
    totals: Dict[int, List[float]] = {skill: [] for skill in range(num_skills)}

    for _ in range(steps):
        skill = int(task_to_skill[int(state.player_state)])
        key, act_key, step_key, reset_key = jax.random.split(key, 4)
        action = int(
            jax.random.randint(
                act_key, shape=(), minval=0, maxval=env.action_space(params).n
            )
        )
        obs, next_state, reward, done, info = env.step(step_key, state, action, params)
        running[skill] += float(reward)

        if bool(info["task_done"]):
            totals[skill].append(running[skill])
            running[skill] = 0.0

        state = next_state

        if bool(done):
            key, reset_key = jax.random.split(reset_key)
            obs, state = env.reset(reset_key, params)

    return totals


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module_path", help="Path to flow module (e.g. exp/.../1.py)")
    parser.add_argument(
        "--env-name",
        default="Craftax-Symbolic-v1",
        help="Craftax flow environment name",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2048,
        help="How many environment steps to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="JAX RNG seed",
    )
    args = parser.parse_args(argv)

    totals = simulate(args.module_path, args.env_name, args.steps, args.seed)

    if not totals:
        print("No skill episodes completed.")
        return

    for skill, rewards in totals.items():
        if not rewards:
            print(f"skill {skill}: n=0")
            continue
        arr = np.array(rewards, dtype=np.float32)
        print(
            f"skill {skill}: n={len(arr)}, mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
