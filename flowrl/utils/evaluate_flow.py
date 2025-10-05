"""
Evaluation utilities for hierarchical flow policies in Craftax.
Designed to run long-term evaluation episodes and calculate comprehensive statistics
at the end of each terminated episode.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import yaml
import importlib.util
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from flowrl.models.actor_critic import ActorCriticMoE, ActorCritic, ActorCriticConv
from flowrl.models.rnd import ActorCriticRND
from flowrl.wrappers import (
    LogWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from flowrl.utils.test import load_policy_params


def convert_jax_arrays_for_json(obj):
    """
    Convert JAX arrays and numpy arrays to lists for JSON serialization.
    Also handles nested dictionaries and lists.
    """
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_jax_arrays_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_jax_arrays_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_jax_arrays_for_json(item) for item in obj]
    else:
        return obj


def save_evaluation_results(
    results: Dict[str, Any], method_name: str, policy_path: str, verbose: bool = True
):
    """
    Save evaluation results to exp_results/method_name/ directory.

    Args:
        results: Dictionary containing evaluation results
        method_name: Name of the method for organizing results
        policy_path: Path to the evaluated policy (for naming)
        verbose: Whether to print save information
    """
    # Create exp_results directory structure
    exp_results_dir = "exp_results"
    method_dir = os.path.join(exp_results_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # Generate filename with timestamp and policy info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_name = os.path.basename(os.path.normpath(policy_path))
    if not policy_name:
        policy_name = "unknown_policy"

    filename = f"{policy_name}_{timestamp}.json"
    filepath = os.path.join(method_dir, filename)

    # Convert JAX arrays to JSON-serializable format
    json_results = convert_jax_arrays_for_json(results)

    # Add metadata
    json_results["_metadata"] = {
        "method_name": method_name,
        "policy_path": policy_path,
        "timestamp": timestamp,
        "evaluation_date": datetime.now().isoformat(),
        "saved_to": filepath,
    }

    # Save to JSON file
    try:
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2, default=str)

        if verbose:
            print(f"\n💾 Results saved to: {filepath}")
            print(f"   Method: {method_name}")
            print(f"   Policy: {policy_name}")
            print(f"   Timestamp: {timestamp}")

        return filepath

    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: Failed to save results to {filepath}: {e}")
        return None


def restore_evaluation_model(policy_path: str, env, num_tasks: int) -> Tuple[Any, Any]:
    """
    Load a trained hierarchical flow policy for evaluation.

    Args:
        policy_path: Path to the trained policy directory
        env: Environment instance
        num_tasks: Number of tasks/skills in the hierarchy

    Returns:
        Tuple of (network, train_state)
    """
    config_path = f"{policy_path}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = {
        k.upper(): v["value"] if type(v) == dict and "value" in v else v
        for k, v in config.items()
    }

    env_params = env.default_params
    network = ActorCriticMoE(
        action_dim=env.action_space(env_params).n,
        layer_width=config["LAYER_SIZE"],
        num_layers=4,
        num_tasks=num_tasks,
    )

    train_state = load_policy_params(policy_path)
    return network, train_state


def restore_ppo_model(
    policy_path: str, env, policy_type: str = "standard"
) -> Tuple[Any, Any]:
    """
    Load a trained PPO policy for evaluation.

    Args:
        policy_path: Path to the trained policy directory
        env: Environment instance
        policy_type: Type of PPO policy ("standard", "rnd", "rnn")

    Returns:
        Tuple of (network, train_state)
    """
    # Try to load config if it exists, otherwise use defaults
    config_path = f"{policy_path}/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = {
            k.upper(): v["value"] if type(v) == dict and "value" in v else v
            for k, v in config.items()
        }
    else:
        # Default config for PPO policies
        config = {
            "LAYER_SIZE": 512,
            "ENV_NAME": "Craftax-Classic-Symbolic-v1",  # Default to classic symbolic
        }

    env_params = env.default_params

    # Create appropriate network based on policy type
    if policy_type == "rnd":
        network = ActorCriticRND(env.action_space(env_params).n, config["LAYER_SIZE"])
    elif policy_type == "rnn":
        # Import here to avoid circular imports
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from ppo_rnn import ActorCriticRNN

        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
    else:  # standard PPO
        if "Symbolic" in config.get("ENV_NAME", ""):
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

    # Load the trained parameters
    loaded_params = load_policy_params(policy_path)

    # Reconstruct TrainState from loaded parameters for PPO
    import optax
    from flax.training.train_state import TrainState

    # Create a dummy optimizer (not used for evaluation)
    tx = optax.adam(1e-4)

    # Create TrainState with loaded parameters
    # Handle nested params structure from checkpoint
    if isinstance(loaded_params, dict):
        if "params" in loaded_params:
            params = loaded_params["params"]
        else:
            params = loaded_params
    else:
        params = loaded_params.params

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    return network, train_state


def setup_evaluation_environment(config: Dict[str, Any], num_envs: int = 32):
    """
    Set up the evaluation environment with proper configuration.

    Args:
        config: Configuration dictionary from policy training
        num_envs: Number of parallel environments for evaluation

    Returns:
        Tuple of (env, env_params, task_to_skill_index, num_tasks, heads, num_heads)
    """
    # Load module if specified
    module_dict = None
    if config.get("MODULE_PATH"):
        module_path = config["MODULE_PATH"]
        module_name = "reward_and_state"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module_dict = module.__dict__

    # Initialize environment
    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_env import make_craftax_flow_env_from_name

        env = make_craftax_flow_env_from_name(
            config["ENV_NAME"],
            not config.get("USE_OPTIMISTIC_RESETS", True),
            module_dict,
        )
        env = LogWrapper(env)
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=num_envs)
    else:
        raise ValueError(f"Unsupported environment: {config['ENV_NAME']}")

    env_params = env.default_params
    task_to_skill_index = jnp.array(env.task_to_skill_index)
    num_tasks = len(task_to_skill_index)
    heads, num_heads = env.heads_info

    return env, env_params, task_to_skill_index, num_tasks, heads, num_heads


def setup_ppo_evaluation_environment(
    env_name: str = "Craftax-Classic-Symbolic-v1", num_envs: int = 32
):
    """
    Set up the evaluation environment for standard PPO policies.

    Args:
        env_name: Name of the environment
        num_envs: Number of parallel environments

    Returns:
        Tuple of (env, env_params)
    """
    # Import here to avoid issues
    from craftax.craftax_env import make_craftax_env_from_name

    # Initialize standard Craftax environment (no flow features)
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    env_params = env.default_params

    return env, env_params


def calculate_episode_statistics(
    trajectory_batch, num_tasks: int, num_achievements: int = 22
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for completed episodes from a trajectory batch.

    Args:
        trajectory_batch: Batch of trajectory data with done flags and info
        num_tasks: Number of hierarchical tasks/states
        num_achievements: Number of achievements (22 for Craftax Classic)

    Returns:
        Dictionary containing episode statistics
    """
    # Get episode termination mask
    done_episodes = trajectory_batch.info[
        "returned_episode"
    ]  # Shape: [timesteps, envs]

    # Only compute statistics for actually terminated episodes
    if done_episodes.sum() == 0:
        # No episodes completed
        return {
            "num_completed_episodes": 0,
            "episode_returns": jnp.array([]),
            "episode_lengths": jnp.array([]),
            "final_state_rates": jnp.zeros(num_tasks),
            "achievement_rates": jnp.zeros(num_achievements),
            "max_state_reached": 0,
        }

    # Calculate basic episode metrics
    episode_returns = trajectory_batch.info["returned_episode_returns"]
    episode_lengths = trajectory_batch.info["returned_episode_lengths"]

    # Filter to only completed episodes
    completed_returns = episode_returns * done_episodes
    completed_lengths = episode_lengths * done_episodes

    # Calculate state visitation rates for completed episodes
    player_state_one_hot = jax.nn.one_hot(
        trajectory_batch.player_state, num_classes=num_tasks
    )

    # Weight by episode completion and sum over time and batch
    weighted_state_visits = player_state_one_hot * done_episodes[:, :, None]
    total_completed_timesteps = done_episodes.sum()
    state_rates = weighted_state_visits.sum(axis=(0, 1)) / jnp.maximum(
        total_completed_timesteps, 1
    )

    # Calculate achievement completion rates
    achievement_rates = jnp.zeros(num_achievements)
    if "reached_state" in trajectory_batch.info:
        # This contains achievement completion information
        reached_info = trajectory_batch.info[
            "reached_state"
        ]  # Shape: [timesteps, envs, achievements]
        weighted_achievements = reached_info * done_episodes[:, :, None]
        achievement_rates = weighted_achievements.sum(axis=(0, 1)) / jnp.maximum(
            total_completed_timesteps, 1
        )

    # Calculate max state reached across all completed episodes
    max_state_reached = jnp.max(trajectory_batch.player_state * done_episodes)

    # Calculate summary statistics
    num_completed_episodes = done_episodes.sum()
    mean_return = completed_returns.sum() / jnp.maximum(num_completed_episodes, 1)
    mean_length = completed_lengths.sum() / jnp.maximum(num_completed_episodes, 1)

    return {
        "num_completed_episodes": num_completed_episodes,
        "mean_episode_return": mean_return,
        "mean_episode_length": mean_length,
        "episode_returns": completed_returns[
            completed_returns > 0
        ],  # Only non-zero returns
        "episode_lengths": completed_lengths[
            completed_lengths > 0
        ],  # Only non-zero lengths
        "final_state_rates": state_rates,
        "achievement_rates": achievement_rates,
        "max_state_reached": max_state_reached,
        "state_progression": jnp.cumsum(
            state_rates
        ),  # Cumulative progression through states
    }


def evaluate_hierarchical_flow_policy(
    policy_path: str,
    max_episode_steps: int = 10000,
    num_envs: int = 32,
    num_evaluation_episodes: int = 1000,
    verbose: bool = True,
    method_name: str = "hierarchical_flow",
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a hierarchical flow policy over many long episodes.

    Args:
        policy_path: Path to trained policy directory
        max_episode_steps: Maximum steps per episode (can be very long)
        num_envs: Number of parallel environments
        num_evaluation_episodes: Total episodes to evaluate across
        verbose: Whether to print progress updates
        method_name: Name of the method for organizing results (default: "hierarchical_flow")
        save_results: Whether to save results to exp_results/ directory (default: True)

    Returns:
        Dictionary containing comprehensive evaluation results
    """
    if verbose:
        print(f"Starting hierarchical flow policy evaluation")
        print(f"Policy: {policy_path}")
        print(f"Max episode steps: {max_episode_steps}")
        print(f"Parallel envs: {num_envs}")
        print(f"Target episodes: {num_evaluation_episodes}")

    # Load configuration
    config_path = f"{policy_path}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = {
        k.upper(): v["value"] if type(v) == dict and "value" in v else v
        for k, v in config.items()
    }

    # Initialize environment and model
    env, env_params, task_to_skill_index, num_tasks, _, num_heads = (
        setup_evaluation_environment(config, num_envs)
    )

    # Override max timesteps for long evaluation episodes
    env_params = env_params.replace(max_timesteps=max_episode_steps)

    network, train_state = restore_evaluation_model(policy_path, env, num_heads)

    if verbose:
        print(f"Loaded model with {num_tasks} tasks and {num_heads} heads")

    # Initialize environment state
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    def map_player_state_to_skill(player_state):
        """Map player states to skill indices for MoE routing."""
        player_state_one_hot = jnp.eye(num_tasks)[player_state]
        player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)
        return player_skill

    @jax.jit
    def evaluation_step(runner_state, _):
        """Single evaluation step function - JIT compiled for performance."""
        train_state, env_state, last_obs, rng = runner_state

        # Select action using policy
        rng, _rng = jax.random.split(rng)
        player_state = env_state.env_state.player_state
        player_skill = map_player_state_to_skill(player_state)

        pi, value = network.apply(train_state["params"], last_obs, player_skill)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # Step environment
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env.step(
            _rng, env_state, action, env_params
        )

        # Create transition for logging
        from flowrl.utils.test import Transition

        transition = Transition(
            done=done,
            player_state=player_state,
            action=action,
            value=value,
            reward=reward,
            reward_i=reward,  # Assuming intrinsic reward = total reward for flow policies
            reward_e=reward,
            log_prob=log_prob,
            obs=last_obs,
            next_obs=new_obs,
            info=info,
        )

        # Update runner_state - preserve hidden state if it exists (for RNN)
        if len(runner_state) == 5:  # RNN case with hidden state
            runner_state = (train_state, new_env_state, new_obs, rng, runner_state[4])
        else:
            runner_state = (train_state, new_env_state, new_obs, rng)

        # Return both new runner_state and the OLD env_state that contains completed episode data
        return runner_state, (transition, env_state)

    # Run evaluation - collect episodes sequentially to avoid bias
    all_episode_returns = []
    all_episode_lengths = []
    all_final_states = []
    all_final_achievements = []  # Store final achievement state for each episode
    all_energy_restored = []  # Track total energy restored per episode
    all_food_restored = []  # Track total food restored per episode
    all_drink_restored = []  # Track total drink restored per episode

    episodes_completed = 0

    while episodes_completed < num_evaluation_episodes:
        print(f"Completed {episodes_completed}/{num_evaluation_episodes} episodes")

        # Reset all environments to start fresh episodes
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        
        # Initialize runner_state - add hidden state for RNN
        if policy_type == "rnn":
            from flowrl.ppo_rnn import ScannedRNN
            hidden_state = ScannedRNN.initialize_carry(num_envs, 512)
            runner_state = (train_state, env_state, obs, rng, hidden_state)
        else:
            runner_state = (train_state, env_state, obs, rng)

        # Run until ALL environments complete their first episode
        episode_done_mask = jnp.zeros(num_envs, dtype=bool)
        step = 0

        # Initialize tracking for restoration counters per environment
        prev_energy = env_state.env_state.player_energy
        prev_food = env_state.env_state.player_food
        prev_drink = env_state.env_state.player_drink
        energy_restored_counts = jnp.zeros(num_envs)
        food_restored_counts = jnp.zeros(num_envs)
        drink_restored_counts = jnp.zeros(num_envs)

        while not jnp.all(episode_done_mask) and step < max_episode_steps:
            # Single evaluation step
            if step % 500 == 0:
                print(f"On step = {step}")
            runner_state, (transition, step_env_state) = evaluation_step(
                runner_state, None
            )

            # Get current resource values from the new environment state
            # Handle both 4-element (standard/RND) and 5-element (RNN) runner_state
            if len(runner_state) == 5:
                _, current_env_state, _, _, _ = runner_state
            else:
                _, current_env_state, _, _ = runner_state
            curr_energy = current_env_state.env_state.player_energy
            curr_food = current_env_state.env_state.player_food
            curr_drink = current_env_state.env_state.player_drink

            # Count restoration (when values go up from previous step)
            energy_increases = jnp.maximum(0, curr_energy - prev_energy)
            food_increases = jnp.maximum(0, curr_food - prev_food)
            drink_increases = jnp.maximum(0, curr_drink - prev_drink)

            # Only count for non-completed episodes (reset when episode completes)
            active_mask = ~episode_done_mask
            energy_restored_counts = jnp.where(
                active_mask,
                energy_restored_counts + energy_increases,
                energy_restored_counts,
            )
            food_restored_counts = jnp.where(
                active_mask, food_restored_counts + food_increases, food_restored_counts
            )
            drink_restored_counts = jnp.where(
                active_mask,
                drink_restored_counts + drink_increases,
                drink_restored_counts,
            )

            # Update previous values for next step
            prev_energy = curr_energy
            prev_food = curr_food
            prev_drink = curr_drink

            # Check which environments just finished their FIRST episode
            new_dones = transition.info["returned_episode"] > 0
            first_completion = new_dones & (~episode_done_mask)
            episode_done_mask = episode_done_mask | new_dones

            # Store results for environments that just completed their first episode
            if jnp.any(first_completion):
                completed_returns = (
                    transition.info["returned_episode_returns"] * first_completion
                )
                completed_lengths = (
                    transition.info["returned_episode_lengths"] * first_completion
                )
                completed_states = transition.player_state * first_completion

                # Get final achievement state for completed episodes from the step's env_state
                final_achievements = step_env_state.env_state.achievements

                # Add to our collections (only non-zero values)
                for i in range(num_envs):
                    if first_completion[i]:
                        all_episode_returns.append(float(completed_returns[i]))
                        all_episode_lengths.append(float(completed_lengths[i]))
                        all_final_states.append(int(completed_states[i]))
                        all_final_achievements.append(
                            final_achievements[i]
                        )  # Store achievement vector
                        all_energy_restored.append(float(energy_restored_counts[i]))
                        all_food_restored.append(float(food_restored_counts[i]))
                        all_drink_restored.append(float(drink_restored_counts[i]))
                        episodes_completed += 1

                        if episodes_completed >= num_evaluation_episodes:
                            break

                # Reset restoration counters for completed episodes
                energy_restored_counts = jnp.where(
                    first_completion, 0, energy_restored_counts
                )
                food_restored_counts = jnp.where(
                    first_completion, 0, food_restored_counts
                )
                drink_restored_counts = jnp.where(
                    first_completion, 0, drink_restored_counts
                )

            step += 1

        # If we still have environments that didn't complete, count them as failed
        if step >= max_episode_steps:
            incomplete_envs = ~episode_done_mask
            if jnp.any(incomplete_envs):
                if verbose:
                    print(
                        f"Warning: {jnp.sum(incomplete_envs)} environments didn't complete within {max_episode_steps} steps"
                    )

                # Add incomplete episodes with current state info
                final_achievements = step_env_state.env_state.achievements
                for i in range(num_envs):
                    if incomplete_envs[i]:
                        all_episode_returns.append(0.0)  # No reward for incomplete
                        all_episode_lengths.append(float(max_episode_steps))
                        all_final_states.append(
                            int(step_env_state.env_state.player_state[i])
                        )
                        all_final_achievements.append(
                            final_achievements[i]
                        )  # Store current achievement state
                        all_energy_restored.append(float(energy_restored_counts[i]))
                        all_food_restored.append(float(food_restored_counts[i]))
                        all_drink_restored.append(float(drink_restored_counts[i]))
                        episodes_completed += 1

                        if episodes_completed >= num_evaluation_episodes:
                            break

    # Aggregate final results
    if not all_episode_returns:
        if verbose:
            print("WARNING: No episodes completed during evaluation!")
        return {
            "total_episodes": 0,
            "mean_episode_return": 0,
            "mean_episode_length": 0,
            "final_state_rates": jnp.zeros(num_tasks),
            "achievement_rates": jnp.zeros(22),
            "max_state_reached": 0,
        }

    # Convert to numpy arrays for easier computation
    all_episode_returns = np.array(all_episode_returns[:num_evaluation_episodes])
    all_episode_lengths = np.array(all_episode_lengths[:num_evaluation_episodes])
    all_final_states = np.array(all_final_states[:num_evaluation_episodes])
    all_final_achievements = jnp.array(
        all_final_achievements[:num_evaluation_episodes]
    )  # Shape: [episodes, achievements]
    all_energy_restored = np.array(all_energy_restored[:num_evaluation_episodes])
    all_food_restored = np.array(all_food_restored[:num_evaluation_episodes])
    all_drink_restored = np.array(all_drink_restored[:num_evaluation_episodes])

    # Calculate state visitation rates
    state_counts = np.bincount(all_final_states, minlength=num_tasks)
    final_state_rates = state_counts / len(all_final_states)

    # Calculate achievement completion counts and rates
    # For each achievement, count how many episodes completed it (achievement value = True)
    achievement_completion_counts = jnp.sum(
        all_final_achievements, axis=0
    )  # Sum across episodes
    achievement_completion_rates = achievement_completion_counts / len(
        all_final_achievements
    )

    # Calculate basic statistics
    mean_return = np.mean(all_episode_returns)
    mean_length = np.mean(all_episode_lengths)
    max_state_reached = np.max(all_final_states)

    # Calculate restoration statistics
    mean_energy_restored = np.mean(all_energy_restored)
    mean_food_restored = np.mean(all_food_restored)
    mean_drink_restored = np.mean(all_drink_restored)
    total_energy_restored = np.sum(all_energy_restored)
    total_food_restored = np.sum(all_food_restored)
    total_drink_restored = np.sum(all_drink_restored)

    final_results = {
        "total_episodes": len(all_episode_returns),
        "mean_episode_return": float(mean_return),
        "mean_episode_length": float(mean_length),
        "final_state_rates": final_state_rates,
        # Achievement statistics - exactly what you wanted
        "achievement_completion_counts": achievement_completion_counts,  # Number of episodes that completed each achievement
        "achievement_completion_rates": achievement_completion_rates,  # Fraction of episodes that completed each achievement
        # Restoration statistics
        "mean_energy_restored": float(mean_energy_restored),
        "mean_food_restored": float(mean_food_restored),
        "mean_drink_restored": float(mean_drink_restored),
        "total_energy_restored": float(total_energy_restored),
        "total_food_restored": float(total_food_restored),
        "total_drink_restored": float(total_drink_restored),
        "max_state_reached": int(max_state_reached),
        "state_progression": np.cumsum(final_state_rates),
        # Raw data for further analysis
        "all_returns": all_episode_returns,
        "all_lengths": all_episode_lengths,
        "all_final_states": all_final_states,
        "all_final_achievements": all_final_achievements,  # Raw achievement data per episode
        "all_energy_restored": all_energy_restored,  # Total energy restored per episode
        "all_food_restored": all_food_restored,  # Total food restored per episode
        "all_drink_restored": all_drink_restored,  # Total drink restored per episode
        # Configuration info
        "policy_path": policy_path,
        "evaluation_config": {
            "max_episode_steps": max_episode_steps,
            "num_envs": num_envs,
            "target_episodes": num_evaluation_episodes,
        },
    }

    if verbose:
        print(f"\n=== Evaluation Results ===")
        print(f"Total episodes completed: {final_results['total_episodes']}")
        print(f"Mean episode return: {final_results['mean_episode_return']:.2f}")
        print(f"Mean episode length: {final_results['mean_episode_length']:.2f}")
        print(f"Max state reached: {final_results['max_state_reached']}")
        print(f"State rates: {final_results['final_state_rates']}")
        print(f"\nRestoration Statistics:")
        print(
            f"  Mean energy restored per episode: {final_results['mean_energy_restored']:.2f}"
        )
        print(
            f"  Mean food restored per episode: {final_results['mean_food_restored']:.2f}"
        )
        print(
            f"  Mean drink restored per episode: {final_results['mean_drink_restored']:.2f}"
        )
        print(f"  Total energy restored: {final_results['total_energy_restored']:.0f}")
        print(f"  Total food restored: {final_results['total_food_restored']:.0f}")
        print(f"  Total drink restored: {final_results['total_drink_restored']:.0f}")

        print(
            f"\nAchievement completion counts (out of {final_results['total_episodes']} episodes):"
        )
        for i, count in enumerate(final_results["achievement_completion_counts"]):
            if count > 0:
                rate = final_results["achievement_completion_rates"][i]
                print(f"  Achievement {i}: {int(count)} episodes ({rate:.2%})")

    # Save results if requested
    if save_results:
        save_evaluation_results(final_results, method_name, policy_path, verbose)

    return final_results


def evaluate_ppo_policy(
    policy_path: str,
    policy_type: str = "standard",
    max_episode_steps: int = 10000,
    num_envs: int = 32,
    num_evaluation_episodes: int = 1000,
    verbose: bool = True,
    method_name: str = "ppo",
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a standard PPO policy over many long episodes.

    Args:
        policy_path: Path to trained policy directory
        policy_type: Type of PPO policy ("standard", "rnd", "rnn")
        max_episode_steps: Maximum steps per episode
        num_envs: Number of parallel environments
        num_evaluation_episodes: Total episodes to evaluate across
        verbose: Whether to print progress updates
        method_name: Name of the method for organizing results
        save_results: Whether to save results to exp_results/ directory

    Returns:
        Dictionary containing comprehensive evaluation results
    """
    if verbose:
        print(f"Starting PPO policy evaluation")
        print(f"Policy: {policy_path}")
        print(f"Policy type: {policy_type}")
        print(f"Max episode steps: {max_episode_steps}")
        print(f"Parallel envs: {num_envs}")
        print(f"Target episodes: {num_evaluation_episodes}")

    # Initialize environment
    # First load config to get environment name
    config_path = f"{policy_path}/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        env_name = config.get("env_name", "Craftax-Classic-Symbolic-v1")
    else:
        env_name = "Craftax-Classic-Symbolic-v1"

    env, env_params = setup_ppo_evaluation_environment(
        env_name=env_name, num_envs=num_envs
    )

    # Override max timesteps for long evaluation episodes
    env_params = env_params.replace(max_timesteps=max_episode_steps)

    # Load model
    network, train_state = restore_ppo_model(policy_path, env, policy_type)

    if verbose:
        print(f"Loaded {policy_type} PPO model")

    # Initialize environment state
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    @jax.jit
    def ppo_evaluation_step(runner_state, _):
        """Single evaluation step function - JIT compiled for performance."""
        if len(runner_state) == 4:
            train_state, env_state, last_obs, rng = runner_state
        else:
            train_state, env_state, last_obs, rng, hidden_state = runner_state

        # Select action using policy
        rng, _rng = jax.random.split(rng)

        if policy_type == "rnd":
            pi, value_e, value_i = network.apply(train_state.params, last_obs)
            value = value_e  # Use extrinsic value for standard metrics
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
        elif policy_type == "rnn":
            # For RNN, we need to handle hidden state properly
            # Initialize hidden state if not present in runner_state

            # For RNN, we need to add sequence dimension and handle dones properly
            # Add sequence dimension of 1 for single timestep evaluation
            obs_seq = last_obs[None, ...]  # Shape: (1, num_envs, obs_dim)
            dones_seq = jnp.zeros((1, num_envs), dtype=bool)  # Shape: (1, num_envs)

            new_hidden, pi, value = network.apply(
                train_state.params, hidden_state, (obs_seq, dones_seq)
            )
            
            # Sample action and then squeeze everything (matching ppo_rnn.py)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0), 
                log_prob.squeeze(0),
            )

            # Update runner_state with new hidden state
            train_state, env_state, last_obs, rng = runner_state[:4]
            runner_state = (train_state, env_state, last_obs, rng, new_hidden)
        else:  # standard PPO
            pi, value = network.apply(train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

        # Step environment
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env.step(
            _rng, env_state, action, env_params
        )

        # Create transition for logging
        from flowrl.utils.test import Transition

        transition = Transition(
            done=done,
            player_state=jnp.zeros(num_envs),  # PPO doesn't have explicit player states
            action=action,
            value=value,
            reward=reward,
            reward_i=jnp.zeros_like(reward),  # No intrinsic reward for standard PPO
            reward_e=reward,
            log_prob=log_prob,
            obs=last_obs,
            next_obs=new_obs,
            info=info,
        )

        # Update runner_state - preserve hidden state if it exists (for RNN)
        if len(runner_state) == 5:  # RNN case with hidden state
            runner_state = (train_state, new_env_state, new_obs, rng, runner_state[4])
        else:
            runner_state = (train_state, new_env_state, new_obs, rng)

        return runner_state, (transition, env_state)

    # Run evaluation - collect episodes sequentially to avoid bias
    all_episode_returns = []
    all_episode_lengths = []
    all_final_achievements = []  # Store final achievement state for each episode
    all_energy_restored = []  # Track total energy restored per episode
    all_food_restored = []  # Track total food restored per episode
    all_drink_restored = []  # Track total drink restored per episode

    episodes_completed = 0

    while episodes_completed < num_evaluation_episodes:
        print(f"Completed {episodes_completed}/{num_evaluation_episodes} episodes")

        # Reset all environments to start fresh episodes
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        
        # Initialize runner_state - add hidden state for RNN
        if policy_type == "rnn":
            from flowrl.ppo_rnn import ScannedRNN
            hidden_state = ScannedRNN.initialize_carry(num_envs, 512)
            runner_state = (train_state, env_state, obs, rng, hidden_state)
        else:
            runner_state = (train_state, env_state, obs, rng)

        # Run until ALL environments complete their first episode
        episode_done_mask = jnp.zeros(num_envs, dtype=bool)
        step = 0

        # Initialize tracking for restoration counters per environment
        prev_energy = env_state.env_state.player_energy
        prev_food = env_state.env_state.player_food
        prev_drink = env_state.env_state.player_drink
        energy_restored_counts = jnp.zeros(num_envs)
        food_restored_counts = jnp.zeros(num_envs)
        drink_restored_counts = jnp.zeros(num_envs)

        while not jnp.all(episode_done_mask) and step < max_episode_steps:
            if step % 50 == 0:
                print(f"Current Step = {step}")
            # Single evaluation step
            runner_state, (transition, step_env_state) = ppo_evaluation_step(
                runner_state, None
            )

            # Store achievement state from the step (for episode completion tracking)
            step_achievements = step_env_state.env_state.achievements

            # Get current resource values from the new environment state
            # Handle both 4-element (standard/RND) and 5-element (RNN) runner_state
            if len(runner_state) == 5:
                _, current_env_state, _, _, _ = runner_state
            else:
                _, current_env_state, _, _ = runner_state
            curr_energy = current_env_state.env_state.player_energy
            curr_food = current_env_state.env_state.player_food
            curr_drink = current_env_state.env_state.player_drink

            # Count restoration (when values go up from previous step)
            energy_increases = jnp.maximum(0, curr_energy - prev_energy)
            food_increases = jnp.maximum(0, curr_food - prev_food)
            drink_increases = jnp.maximum(0, curr_drink - prev_drink)

            # Only count for non-completed episodes
            active_mask = ~episode_done_mask
            energy_restored_counts = jnp.where(
                active_mask,
                energy_restored_counts + energy_increases,
                energy_restored_counts,
            )
            food_restored_counts = jnp.where(
                active_mask, food_restored_counts + food_increases, food_restored_counts
            )
            drink_restored_counts = jnp.where(
                active_mask,
                drink_restored_counts + drink_increases,
                drink_restored_counts,
            )

            # Update previous values for next step
            prev_energy = curr_energy
            prev_food = curr_food
            prev_drink = curr_drink

            # Check which environments just finished their FIRST episode
            new_dones = transition.info["returned_episode"] > 0
            first_completion = new_dones & (~episode_done_mask)
            episode_done_mask = episode_done_mask | new_dones

            # Store results for environments that just completed their first episode
            if jnp.any(first_completion):
                completed_returns = (
                    transition.info["returned_episode_returns"] * first_completion
                )
                completed_lengths = (
                    transition.info["returned_episode_lengths"] * first_completion
                )

                # Add to our collections (only non-zero values)
                # Extract achievements from environment state (from step_env_state)
                final_achievements = step_achievements
                for i in range(num_envs):
                    if first_completion[i]:
                        all_episode_returns.append(float(completed_returns[i]))
                        all_episode_lengths.append(float(completed_lengths[i]))
                        all_final_achievements.append(final_achievements[i])
                        all_energy_restored.append(float(energy_restored_counts[i]))
                        all_food_restored.append(float(food_restored_counts[i]))
                        all_drink_restored.append(float(drink_restored_counts[i]))
                        episodes_completed += 1

                        if episodes_completed >= num_evaluation_episodes:
                            break

                # Reset restoration counters for completed episodes
                energy_restored_counts = jnp.where(
                    first_completion, 0, energy_restored_counts
                )
                food_restored_counts = jnp.where(
                    first_completion, 0, food_restored_counts
                )
                drink_restored_counts = jnp.where(
                    first_completion, 0, drink_restored_counts
                )

            step += 1

        # If we still have environments that didn't complete, count them as failed
        if step >= max_episode_steps:
            incomplete_envs = ~episode_done_mask
            if jnp.any(incomplete_envs):
                if verbose:
                    print(
                        f"Warning: {jnp.sum(incomplete_envs)} environments didn't complete within {max_episode_steps} steps"
                    )

                # Add incomplete episodes with current state info
                final_achievements = step_achievements
                for i in range(num_envs):
                    if incomplete_envs[i]:
                        all_episode_returns.append(0.0)  # No reward for incomplete
                        all_episode_lengths.append(float(max_episode_steps))
                        all_final_achievements.append(final_achievements[i])
                        all_energy_restored.append(float(energy_restored_counts[i]))
                        all_food_restored.append(float(food_restored_counts[i]))
                        all_drink_restored.append(float(drink_restored_counts[i]))
                        episodes_completed += 1

                        if episodes_completed >= num_evaluation_episodes:
                            break

    # Aggregate final results
    if not all_episode_returns:
        if verbose:
            print("WARNING: No episodes completed during evaluation!")
        return {
            "total_episodes": 0,
            "mean_episode_return": 0,
            "mean_episode_length": 0,
        }

    # Convert to numpy arrays for easier computation
    all_episode_returns = np.array(all_episode_returns[:num_evaluation_episodes])
    all_episode_lengths = np.array(all_episode_lengths[:num_evaluation_episodes])
    all_final_achievements = jnp.array(
        all_final_achievements[:num_evaluation_episodes]
    )  # Shape: [episodes, achievements]
    all_energy_restored = np.array(all_energy_restored[:num_evaluation_episodes])
    all_food_restored = np.array(all_food_restored[:num_evaluation_episodes])
    all_drink_restored = np.array(all_drink_restored[:num_evaluation_episodes])

    # Calculate basic statistics
    mean_return = np.mean(all_episode_returns)
    mean_length = np.mean(all_episode_lengths)
    
    # Calculate achievement completion counts and rates
    achievement_completion_counts = jnp.sum(
        all_final_achievements, axis=0
    )  # Sum across episodes
    achievement_completion_rates = achievement_completion_counts / len(
        all_final_achievements
    )

    # Calculate restoration statistics
    mean_energy_restored = np.mean(all_energy_restored)
    mean_food_restored = np.mean(all_food_restored)
    mean_drink_restored = np.mean(all_drink_restored)
    total_energy_restored = np.sum(all_energy_restored)
    total_food_restored = np.sum(all_food_restored)
    total_drink_restored = np.sum(all_drink_restored)

    final_results = {
        "total_episodes": len(all_episode_returns),
        "mean_episode_return": float(mean_return),
        "mean_episode_length": float(mean_length),
        # Achievement statistics
        "achievement_completion_counts": achievement_completion_counts,
        "achievement_completion_rates": achievement_completion_rates,
        # Restoration statistics
        "mean_energy_restored": float(mean_energy_restored),
        "mean_food_restored": float(mean_food_restored),
        "mean_drink_restored": float(mean_drink_restored),
        "total_energy_restored": float(total_energy_restored),
        "total_food_restored": float(total_food_restored),
        "total_drink_restored": float(total_drink_restored),
        # Raw data for further analysis
        "all_returns": all_episode_returns,
        "all_lengths": all_episode_lengths,
        "all_final_achievements": all_final_achievements,  # Raw achievement data per episode
        "all_energy_restored": all_energy_restored,  # Total energy restored per episode
        "all_food_restored": all_food_restored,  # Total food restored per episode
        "all_drink_restored": all_drink_restored,  # Total drink restored per episode
        # Configuration info
        "policy_path": policy_path,
        "policy_type": policy_type,
        "evaluation_config": {
            "max_episode_steps": max_episode_steps,
            "num_envs": num_envs,
            "target_episodes": num_evaluation_episodes,
        },
    }

    if verbose:
        print(f"\n=== PPO Evaluation Results ===")
        print(f"Total episodes completed: {final_results['total_episodes']}")
        print(f"Mean episode return: {final_results['mean_episode_return']:.2f}")
        print(f"Mean episode length: {final_results['mean_episode_length']:.2f}")
        print(f"\nRestoration Statistics:")
        print(
            f"  Mean energy restored per episode: {final_results['mean_energy_restored']:.2f}"
        )
        print(
            f"  Mean food restored per episode: {final_results['mean_food_restored']:.2f}"
        )
        print(
            f"  Mean drink restored per episode: {final_results['mean_drink_restored']:.2f}"
        )
        print(f"  Total energy restored: {final_results['total_energy_restored']:.0f}")
        print(f"  Total food restored: {final_results['total_food_restored']:.0f}")
        print(f"  Total drink restored: {final_results['total_drink_restored']:.0f}")
        print(
            f"\nAchievement completion counts (out of {final_results['total_episodes']} episodes):"
        )
        for i, count in enumerate(final_results["achievement_completion_counts"]):
            if count > 0:
                rate = final_results["achievement_completion_rates"][i]
                print(f"  Achievement {i}: {int(count)} episodes ({rate:.2%})")

    # Save results if requested
    if save_results:
        save_evaluation_results(final_results, method_name, policy_path, verbose)

    return final_results


def compare_hierarchical_policies(
    policy_paths: List[str],
    max_episode_steps: int = 10000,
    num_envs: int = 32,
    num_evaluation_episodes: int = 500,
    save_results: bool = True,
    results_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare multiple hierarchical flow policies side-by-side.

    Args:
        policy_paths: List of paths to trained policies
        max_episode_steps: Maximum steps per episode
        num_envs: Number of parallel environments
        num_evaluation_episodes: Episodes per policy
        save_results: Whether to save comparison results
        results_path: Where to save results (optional)

    Returns:
        Dictionary containing comparison results
    """
    results = {}

    print(f"Comparing {len(policy_paths)} hierarchical policies...")

    for i, policy_path in enumerate(policy_paths):
        print(f"\n--- Evaluating Policy {i+1}/{len(policy_paths)}: {policy_path} ---")

        try:
            policy_results = evaluate_hierarchical_flow_policy(
                policy_path=policy_path,
                max_episode_steps=max_episode_steps,
                num_envs=num_envs,
                num_evaluation_episodes=num_evaluation_episodes,
                verbose=True,
            )
            results[policy_path] = policy_results

        except Exception as e:
            print(f"ERROR evaluating {policy_path}: {e}")
            results[policy_path] = {"error": str(e)}

    # Create comparison summary
    comparison = {
        "policies": policy_paths,
        "individual_results": results,
        "comparison_summary": {},
    }

    # Extract key metrics for easy comparison
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if valid_results:
        comparison["comparison_summary"] = {
            "mean_returns": {
                k: v["mean_episode_return"] for k, v in valid_results.items()
            },
            "mean_lengths": {
                k: v["mean_episode_length"] for k, v in valid_results.items()
            },
            "max_states": {k: v["max_state_reached"] for k, v in valid_results.items()},
            "total_episodes": {
                k: v["total_episodes"] for k, v in valid_results.items()
            },
        }

        # Find best performing policy
        best_return_policy = max(
            valid_results.keys(), key=lambda x: valid_results[x]["mean_episode_return"]
        )
        best_state_policy = max(
            valid_results.keys(), key=lambda x: valid_results[x]["max_state_reached"]
        )

        comparison["comparison_summary"]["best_return_policy"] = best_return_policy
        comparison["comparison_summary"]["best_state_policy"] = best_state_policy

        print(f"\n=== Comparison Summary ===")
        print(f"Best return policy: {best_return_policy}")
        print(f"Best state progression policy: {best_state_policy}")

    # Save results if requested
    if save_results:
        import json

        save_path = results_path or "hierarchical_policy_comparison.json"

        # Convert JAX arrays to lists for JSON serialization
        def convert_jax_arrays(obj):
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_jax_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_jax_arrays(item) for item in obj]
            else:
                return obj

        json_results = convert_jax_arrays(comparison)

        with open(save_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {save_path}")

    return comparison


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate hierarchical flow policies")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to trained policy directory",
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=10000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--num_envs", type=int, default=32, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to JSON file"
    )

    args = parser.parse_args()

    results = evaluate_hierarchical_flow_policy(
        policy_path=args.policy_path,
        max_episode_steps=args.max_episode_steps,
        num_envs=args.num_envs,
        num_evaluation_episodes=args.num_episodes,
        verbose=True,
    )

    if args.save_results:
        import json

        def convert_jax_arrays(obj):
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_jax_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_jax_arrays(item) for item in obj]
            else:
                return obj

        json_results = convert_jax_arrays(results)
        save_path = f"evaluation_results_{os.path.basename(args.policy_path)}.json"

        with open(save_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {save_path}")
