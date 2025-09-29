#!/usr/bin/env python3
"""
Test script for policy evaluation system.
Supports both hierarchical flow policies and standard PPO policies.
"""

import argparse
import os
from flowrl.utils.evaluate_flow import (
    evaluate_hierarchical_flow_policy,
    evaluate_ppo_policy,
)


def evaluate_policy(
    policy_directory: str,
    is_flow_policy: bool = True,
    policy_type: str = "standard",
    method_name: str = None,
    max_episode_steps: int = 400,
    num_envs: int = 32,
    num_evaluation_episodes: int = 100,
    verbose: bool = True,
    save_results: bool = True,
):
    """
    Evaluate a trained policy.

    Args:
        policy_directory: Directory containing the trained policy
        is_flow_policy: Whether this is a hierarchical flow policy (True) or standard PPO (False)
        policy_type: Type of PPO policy ("standard", "rnd", "rnn") - only used for non-flow policies
        method_name: Name of the method for organizing results (auto-determined if None)
        max_episode_steps: Maximum steps per episode
        num_envs: Number of parallel environments
        num_evaluation_episodes: Total number of episodes to evaluate
        verbose: Whether to print detailed output
        save_results: Whether to save results to exp_results/ directory

    Returns:
        Dictionary containing evaluation results
    """
    # Auto-determine method name if not provided
    if method_name is None:
        if is_flow_policy:
            method_name = "hierarchical_flow"
        else:
            method_name = f"ppo_{policy_type}"

    if is_flow_policy:
        print("Evaluating hierarchical flow policy...")
        results = evaluate_hierarchical_flow_policy(
            policy_path=policy_directory,
            max_episode_steps=max_episode_steps,
            num_envs=num_envs,
            num_evaluation_episodes=num_evaluation_episodes,
            verbose=verbose,
            method_name=method_name,
            save_results=save_results,
        )
    else:
        print(f"Evaluating {policy_type} PPO policy...")
        results = evaluate_ppo_policy(
            policy_path=policy_directory,
            policy_type=policy_type,
            max_episode_steps=max_episode_steps,
            num_envs=num_envs,
            num_evaluation_episodes=num_evaluation_episodes,
            verbose=verbose,
            method_name=method_name,
            save_results=save_results,
        )

    return results


def print_evaluation_results(results, is_flow_policy: bool):
    """Print formatted evaluation results."""
    print("Evaluation completed!")
    print(f"Results keys: {list(results.keys())}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Mean return: {results['mean_episode_return']:.2f}")
    print(f"Mean length: {results['mean_episode_length']:.2f}")

    if is_flow_policy:
        print(f"Max state reached: {results['max_state_reached']}")

        print(f"\n=== Achievement Analysis ===")
        print(
            f"Achievement completion counts: {results['achievement_completion_counts']}"
        )
        print(
            f"Achievement completion rates: {results['achievement_completion_rates']}"
        )

        # Show which achievements were completed most frequently
        completed_achievements = [
            (i, int(count))
            for i, count in enumerate(results["achievement_completion_counts"])
            if count > 0
        ]
        if completed_achievements:
            completed_achievements.sort(key=lambda x: x[1], reverse=True)
            print(f"\nMost frequently completed achievements:")
            for achievement_id, count in completed_achievements[:10]:  # Top 10
                rate = results["achievement_completion_rates"][achievement_id]
                print(
                    f"  Achievement {achievement_id}: {count}/{results['total_episodes']} episodes ({rate:.1%})"
                )
        else:
            print("No achievements were completed in any episodes.")

    # Print restoration statistics if available
    if "mean_energy_restored" in results:
        print(f"\n=== Restoration Statistics ===")
        print(
            f"Mean energy restored per episode: {results['mean_energy_restored']:.2f}"
        )
        print(f"Mean food restored per episode: {results['mean_food_restored']:.2f}")
        print(f"Mean drink restored per episode: {results['mean_drink_restored']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument(
        "policy_directory", type=str, help="Directory containing the trained policy"
    )
    parser.add_argument(
        "--flow_policy",
        action="store_true",
        default=True,
        help="Whether this is a hierarchical flow policy (default: True)",
    )
    parser.add_argument(
        "--standard_ppo",
        action="store_true",
        help="Whether this is a standard PPO policy (overrides --flow_policy)",
    )
    parser.add_argument(
        "--ppo_type",
        type=str,
        choices=["standard", "rnd", "rnn"],
        default="standard",
        help="Type of PPO policy for non-flow policies (default: standard)",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="Name of the method for organizing results (auto-determined if not provided)",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=10000,
        help="Maximum steps per episode (default: 5000)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
        help="Number of parallel environments (default: 32)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10000,
        help="Number of episodes to evaluate (default: 100)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to exp_results/ directory",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate policy directory exists
    if not os.path.exists(args.policy_directory):
        print(f"❌ Error: Policy directory '{args.policy_directory}' does not exist")
        exit(1)

    # Determine policy type
    is_flow_policy = args.flow_policy and not args.standard_ppo
    save_results = not args.no_save

    if is_flow_policy:
        policy_type_str = "Hierarchical Flow"
    else:
        policy_type_str = f"{args.ppo_type.upper()} PPO"

    print(f"=== Evaluating {policy_type_str} Policy ===")
    print(f"Policy directory: {args.policy_directory}")
    if args.method_name:
        print(f"Method name: {args.method_name}")
    print(f"Save results: {'Yes' if save_results else 'No'}")

    try:
        results = evaluate_policy(
            policy_directory=args.policy_directory,
            is_flow_policy=is_flow_policy,
            policy_type=args.ppo_type,
            method_name=args.method_name,
            max_episode_steps=args.max_episode_steps,
            num_envs=args.num_envs,
            num_evaluation_episodes=args.num_episodes,
            verbose=not args.quiet,
            save_results=save_results,
        )

        if not args.quiet:
            print_evaluation_results(results, is_flow_policy)

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
