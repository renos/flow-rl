#!/usr/bin/env python3
"""Compare JAX compile times for standard PPO and Flow MoE networks."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import tree_util

# Ensure the bundled Craftax package is importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent
CRAFTAX_PATH = REPO_ROOT / "Craftax"
if str(CRAFTAX_PATH) not in sys.path:
    sys.path.insert(0, str(CRAFTAX_PATH))

from craftax.craftax_env import make_craftax_flow_env_from_name
from flowrl.models.actor_critic import ActorCritic, ActorCriticMoE


def load_module_dict(module_path: Path) -> dict[str, object]:
    """Load the generated reward/state module used to configure Flow tasks."""
    spec = importlib.util.spec_from_file_location("reward_and_state", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def block_until_ready(tree):
    """Synchronise on all array leaves in a pytree."""
    for leaf in tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            jax.block_until_ready(leaf)


def measure_compile_time(fn, *args) -> float:
    """Invoke `fn(*args)` once and return the wall-clock duration."""
    start = time.perf_counter()
    outputs = fn(*args)
    block_until_ready(outputs)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compile times for Flow PPO networks (standard vs MoE)."
    )
    parser.add_argument(
        "--module",
        type=Path,
        required=True,
        help="Path to the generated reward_and_state module (e.g. exp/premade/1.py).",
    )
    parser.add_argument(
        "--env",
        default="Craftax-Classic-Symbolic-v1",
        help="Environment name for make_craftax_flow_env_from_name.",
    )
    parser.add_argument(
        "--layer-width",
        type=int,
        default=512,
        help="Hidden layer width shared by both networks.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Depth for the MoE network (matches PPO Flow defaults).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dummy inputs used during compilation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for parameter initialisation.",
    )
    args = parser.parse_args()

    module_path = args.module.resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module path does not exist: {module_path}")

    module_dict = load_module_dict(module_path)

    env = make_craftax_flow_env_from_name(args.env, False, module_dict)
    env_params = env.default_params
    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    _, num_heads = env.heads_info
    if num_heads <= 0:
        raise ValueError("Environment reported zero MoE heads.")

    rng = jax.random.PRNGKey(args.seed)
    rng, normal_key, moe_key = jax.random.split(rng, 3)

    obs = jnp.zeros((args.batch_size, *obs_shape), dtype=jnp.float32)
    task_ids = jnp.zeros((args.batch_size,), dtype=jnp.int32)

    normal_network = ActorCritic(action_dim=action_dim, layer_width=args.layer_width)
    moe_network = ActorCriticMoE(
        action_dim=action_dim,
        layer_width=args.layer_width,
        num_layers=args.num_layers,
        num_tasks=num_heads,
    )

    normal_params = normal_network.init(normal_key, obs)
    moe_params = moe_network.init(moe_key, obs, task_ids)

    @jax.jit
    def normal_forward(params, inputs):
        pi, value = normal_network.apply(params, inputs)
        return pi.logits, value

    @jax.jit
    def moe_forward(params, inputs, skills):
        pi, value = moe_network.apply(params, inputs, skills)
        return pi.logits, value

    normal_time = measure_compile_time(normal_forward, normal_params, obs)
    moe_time = measure_compile_time(moe_forward, moe_params, obs, task_ids)

    delta = moe_time - normal_time
    relative = (delta / normal_time * 100.0) if normal_time > 0 else float("nan")

    print("=== Compile Time Comparison ===")
    print(f"Environment: {args.env}")
    print(f"Reward module: {module_path}")
    print(f"Action dimension: {action_dim}")
    print(f"Observation shape: {obs_shape}")
    print(f"MoE heads: {num_heads}")
    print(f"Batch size: {args.batch_size}")
    print()
    print(f"ActorCritic compile time:  {normal_time:.3f} s")
    print(f"ActorCriticMoE compile time: {moe_time:.3f} s")
    print(f"Difference (MoE - base):   {delta:.3f} s ({relative:.1f}%)")


if __name__ == "__main__":
    main()
