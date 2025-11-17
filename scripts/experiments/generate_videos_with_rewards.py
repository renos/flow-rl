#!/usr/bin/env python3
"""
Generate example videos from hierarchical flow policies with reward information displayed.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import yaml
import importlib.util
import cv2
from typing import Tuple

from flowrl.utils.test import restore_model, load_policy_params
from flowrl.wrappers import AutoResetEnvWrapper, BatchEnvWrapper
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax.renderer import render_craftax_pixels as render_craftax_pixels_symbolic
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN as CRAFTAX_BLOCK_PIXEL_SIZE_HUMAN
# Fabrax imports
from craftax.fabrax.renderer import render_craftax_pixels as render_fabrax_pixels
from craftax.fabrax.constants import BLOCK_PIXEL_SIZE_HUMAN as FABRAX_BLOCK_PIXEL_SIZE_HUMAN
# Regular environment imports
from craftax.craftax_env import make_craftax_env_from_name
from flowrl.ppo_rnn import ActorCriticRNN, ScannedRNN
from flax.training.train_state import TrainState
import optax


def gen_frames_ppo_with_rewards(policy_path, max_num_frames=2000, num_envs=128, env_name=None):
    """
    Generate frames with reward information from a regular PPO policy (non-hierarchical).

    Returns:
        frames, states, env_states, actions, rewards for a random trajectory
    """
    # Try to load config if it exists, otherwise use defaults (like in evaluate_flow.py)
    config_path = f"{policy_path}/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = {
            k.upper(): v["value"] if type(v) == dict and "value" in v else v
            for k, v in config_.items()
        }
    else:
        # Default config for PPO policies
        config = {
            "LAYER_SIZE": 512,
            "ENV_NAME": env_name or "Craftax-Classic-Symbolic-v1",  # Use passed env_name or default to classic symbolic
            "USE_OPTIMISTIC_RESETS": True,
        }

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)

    # Set up regular environment (use same approach as evaluate_flow.py)
    env_name = config["ENV_NAME"]
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)
    env_params = env.default_params

    # Load model using the same approach as evaluate_flow.py
    print(f"Loading PPO model from: {policy_path}")
    from flowrl.models.actor_critic import ActorCritic, ActorCriticConv
    from flowrl.utils.test import load_policy_params
    import optax
    from flax.training.train_state import TrainState

    # Create appropriate network based on environment type
    if "Symbolic" in env_name:
        network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
    else:
        network = ActorCriticConv(env.action_space(env_params).n, config["LAYER_SIZE"])

    # Load the trained parameters
    loaded_params = load_policy_params(policy_path)

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

    def batch_step_fn(carry, x):
        rng, obs, env_state = carry
        rng, _rng = jax.random.split(rng)

        # For regular PPO, no player skill mapping needed
        pi, value = network.apply(train_state.params, obs)
        action = pi.sample(seed=_rng)

        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env.step(
            _rng, env_state, action, env_params
        )

        return (rng, new_obs, new_env_state), (new_env_state, action, reward, _rng)

    # JIT compile the batch rollout
    @jax.jit
    def run_batch_rollout(initial_carry, inputs):
        return jax.lax.scan(batch_step_fn, initial_carry, inputs)

    # Run a single rollout (no goal state needed for regular PPO)
    print(f"Running rollout with {num_envs} parallel environments...")

    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)
    initial_carry = (rng, obs, env_state)

    inputs = jnp.arange(max_num_frames)
    _, (env_states, actions, rewards, rngs) = run_batch_rollout(initial_carry, inputs)

    # Select a random environment for video generation
    env_idx = np.random.randint(0, num_envs)
    print(f"Using environment {env_idx} for video generation")

    # Extract trajectory from selected environment
    def extract_env(x):
        return x[:, env_idx] if x.ndim > 1 else x[env_idx]

    selected_env_states = jax.tree.map(extract_env, env_states)
    selected_actions = actions[:, env_idx]
    selected_rewards = rewards[:, env_idx]

    # Render the trajectory with reward information
    print("Rendering trajectory frames with rewards...")

    # Select appropriate renderer based on environment
    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        renderer = jax.jit(render_craftax_pixels, static_argnums=(1,))
        block_pixel_size = BLOCK_PIXEL_SIZE_HUMAN
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        # Disable night noise for clarity; call positionally to avoid kwarg issues on lambdas
        renderer = jax.jit(lambda s, bps: render_craftax_pixels_symbolic(s, bps, False), static_argnums=(1,))
        block_pixel_size = CRAFTAX_BLOCK_PIXEL_SIZE_HUMAN
    elif config["ENV_NAME"] == "Fabrax-Symbolic-v1":
        renderer = jax.jit(render_fabrax_pixels, static_argnums=(1,))
        block_pixel_size = FABRAX_BLOCK_PIXEL_SIZE_HUMAN

    # Render each environment state directly
    def render_state_with_reward(env_state, reward, step):
        # Pass block size positionally to support lambda wrappers
        frame = renderer(env_state, block_pixel_size)
        # Upscale to ~64px tiles for readability
        scale = max(1, 64 // int(block_pixel_size))
        frame_rgb = np.array(frame, dtype=np.float32)
        if scale > 1:
            frame_rgb = np.repeat(np.repeat(frame_rgb, scale, axis=0), scale, axis=1)
        # Renderer outputs 0..255 floats; convert to uint8
        frame_with_text = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        # Add reward text (top left)
        cv2.putText(
            frame_with_text,
            f"Reward: {float(reward):.3f}",
            (10, 30),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Font scale
            (255, 255, 255),  # Color (white)
            2,  # Thickness
        )

        # Add step text (top left, below reward)
        cv2.putText(
            frame_with_text,
            f"Step: {int(step)}",
            (10, 60),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Font scale
            (255, 255, 255),  # Color (white)
            2,  # Thickness
        )

        return frame_with_text

    # Render all frames with reward information
    num_frames = len(selected_rewards)
    print(f"Rendering {num_frames} frames with reward information")

    frames_list = []
    for i in range(num_frames):
        # Extract single frame data
        env_state_i = jax.tree.map(lambda x: x[i], selected_env_states)
        reward_i = selected_rewards[i]

        # Render frame with reward info
        frame_with_info = render_state_with_reward(env_state_i, reward_i, i)
        frames_list.append(frame_with_info)

        if i % 100 == 0:
            print(f"  Rendered frame {i}/{num_frames}")

    frames = np.array(frames_list)

    # Create dummy states array (no hierarchical states for regular PPO)
    states = np.arange(len(frames))  # Just use frame indices
    env_states = selected_env_states
    actions = selected_actions
    rewards = selected_rewards

    print(f"Generated {len(frames)} frames for trajectory")
    print(f"Total reward: {float(jnp.sum(rewards)):.3f}")
    print(f"Average reward per step: {float(jnp.mean(rewards)):.3f}")

    return frames, states, env_states, actions, rewards


def gen_frames_ppo_rnn_with_rewards(policy_path, max_num_frames=2000, num_envs=128, env_name=None):
    """
    Generate frames with reward information from a PPO-RNN policy.

    Returns:
        frames, states, env_states, actions, rewards for a random trajectory
    """
    # Load config if available
    config_path = f"{policy_path}/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = {k.upper(): v["value"] if type(v) == dict and "value" in v else v for k, v in config_.items()}
        env_name = config.get("ENV_NAME", env_name or "Craftax-Symbolic-v1")
        layer_size = int(config.get("LAYER_SIZE", 512))
    else:
        env_name = env_name or "Craftax-Symbolic-v1"
        layer_size = 512

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)

    # Set up environment (batched)
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)
    env_params = env.default_params

    # Build RNN network and load params
    network = ActorCriticRNN(env.action_space(env_params).n, config={"LAYER_SIZE": layer_size})
    loaded_params = load_policy_params(policy_path)
    params = loaded_params["params"] if isinstance(loaded_params, dict) and "params" in loaded_params else loaded_params.params
    tx = optax.adam(1e-4)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    @jax.jit
    def batch_step_fn(carry, _):
        rng, obs, env_state, hidden = carry
        rng, _rng = jax.random.split(rng)

        # Add sequence dim for one-step RNN call
        obs_seq = obs[None, ...]
        dones_seq = jnp.zeros((1, obs.shape[0]), dtype=bool)
        new_hidden, pi, value = network.apply(train_state.params, hidden, (obs_seq, dones_seq))
        action = pi.sample(seed=_rng)
        action = jnp.squeeze(action, axis=0)  # remove sequence dim -> [num_envs]

        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        return (rng, new_obs, new_env_state, new_hidden), (new_env_state, action, reward, _rng)

    # Reset and rollout
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)
    hidden = ScannedRNN.initialize_carry(num_envs, layer_size)
    initial_carry = (rng, obs, env_state, hidden)
    inputs = jnp.arange(max_num_frames)
    (_, _, env_states, _), (env_states_hist, actions, rewards, rngs) = jax.lax.scan(batch_step_fn, initial_carry, inputs)

    # Pick a random env to render
    env_idx = np.random.randint(0, num_envs)
    print(f"Using environment {env_idx} for video generation (RNN)")

    def extract_env(x):
        return x[:, env_idx] if x.ndim > 1 else x[env_idx]

    selected_env_states = jax.tree.map(extract_env, env_states_hist)
    selected_actions = actions[:, env_idx]
    selected_rewards = rewards[:, env_idx]

    # Select renderer
    if env_name == "Craftax-Classic-Symbolic-v1":
        renderer = jax.jit(render_craftax_pixels, static_argnums=(1,))
        block_pixel_size = BLOCK_PIXEL_SIZE_HUMAN
    elif env_name == "Craftax-Symbolic-v1":
        renderer = jax.jit(render_craftax_pixels_symbolic, static_argnums=(1,))
        block_pixel_size = CRAFTAX_BLOCK_PIXEL_SIZE_HUMAN
    elif env_name == "Fabrax-Symbolic-v1":
        renderer = jax.jit(render_fabrax_pixels, static_argnums=(1,))
        block_pixel_size = FABRAX_BLOCK_PIXEL_SIZE_HUMAN
    else:
        renderer = jax.jit(render_craftax_pixels_symbolic, static_argnums=(1,))
        block_pixel_size = CRAFTAX_BLOCK_PIXEL_SIZE_HUMAN

    # Render frames with reward text similar to non-RNN variant
    frames_list = []
    for i in range(len(selected_rewards)):
        env_state_i = jax.tree.map(lambda x: x[i], selected_env_states)
        reward_i = selected_rewards[i]
        frame = renderer(env_state_i, block_pixel_size)
        frame_with_text = np.array(frame, dtype=np.uint8)
        cv2.putText(frame_with_text, f"Reward: {float(reward_i):.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame_with_text, f"Step: {int(i)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        frames_list.append(frame_with_text)

    frames = np.array(frames_list)
    states = np.arange(len(frames))
    env_states = selected_env_states
    actions = selected_actions
    rewards = selected_rewards
    print(f"Generated {len(frames)} RNN frames; total reward {float(jnp.sum(rewards)):.3f}")
    return frames, states, env_states, actions, rewards


def gen_frames_hierarchical_with_rewards(policy_path, max_num_frames=2000, goal_state=None, num_envs=128, require_goal_state=True):
    """
    Generate frames with reward information for each timestep.
    
    Returns:
        frames, states, env_states, actions, rewards for the successful trajectory
    """
    config_path = f"{policy_path}/config.yaml"
    with open(config_path, "r") as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
    config = {
        k.upper(): v["value"] if type(v) == dict and "value" in v else v
        for k, v in config_.items()
    }

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    
    # Set up batched environment to find successful trajectory
    if config["MODULE_PATH"]:
        module_path = config["MODULE_PATH"]
        module_name = "reward_and_state"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module_dict = module.__dict__
    else:
        module_dict = None

    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_env import make_craftax_flow_env_from_name

        # Batched environment for finding successful trajectory
        env_batch = make_craftax_flow_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"], module_dict
        )
        env_batch = AutoResetEnvWrapper(env_batch)
        env_batch = BatchEnvWrapper(env_batch, num_envs=num_envs)
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax_env import make_craftax_flow_env_from_name

        # Batched environment for finding successful trajectory (Craftax symbolic)
        env_batch = make_craftax_flow_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"], module_dict
        )
        env_batch = AutoResetEnvWrapper(env_batch)
        env_batch = BatchEnvWrapper(env_batch, num_envs=num_envs)
    elif config["ENV_NAME"] == "Fabrax-Symbolic-v1":
        from craftax.craftax_env import make_craftax_flow_env_from_name

        # Batched environment for finding successful trajectory
        env_batch = make_craftax_flow_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"], module_dict
        )
        env_batch = AutoResetEnvWrapper(env_batch)
        env_batch = BatchEnvWrapper(env_batch, num_envs=num_envs)
    else:
        raise ValueError(f"Unsupported environment: {config['ENV_NAME']}. Supported environments: Craftax-Classic-Symbolic-v1, Fabrax-Symbolic-v1")

    env_params = env_batch.default_params
    task_to_skill_index = jnp.array(env_batch.task_to_skill_index)
    num_tasks_ = len(task_to_skill_index)
    heads, num_heads = env_batch.heads_info

    # Load model
    print(f"Finding successful trajectory with {num_envs} parallel environments...")
    network, train_state = restore_model(policy_path, env_batch, num_heads)

    def map_player_state_to_skill(player_state):
        player_state_one_hot = jnp.eye(num_tasks_)[player_state]
        player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)
        return player_skill

    def batch_step_fn(carry, x):
        rng, obs, env_state = carry
        rng, _rng = jax.random.split(rng)

        player_state = env_state.player_state
        player_skill = map_player_state_to_skill(player_state)

        pi, value = network.apply(train_state["params"], obs, player_skill)
        action = pi.sample(seed=_rng)

        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env_batch.step(
            _rng, env_state, action, env_params
        )

        return (rng, new_obs, new_env_state), (new_env_state, action, reward, _rng)
    
    # JIT compile the batch rollout for maximum performance
    @jax.jit
    def run_batch_rollout(initial_carry, inputs):
        return jax.lax.scan(batch_step_fn, initial_carry, inputs)

    # Simple recording path: skip goal search and stream one-env rendering
    if not require_goal_state:
        print("Simple mode: recording a trajectory without goal search")

        # Choose a small batch to keep memory low and a single env to render
        render_env_idx = 0

        # Select renderer based on environment and prepare pixel size
        if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
            renderer = jax.jit(render_craftax_pixels, static_argnums=(1,))
            block_pixel_size = BLOCK_PIXEL_SIZE_HUMAN
        elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
            renderer = jax.jit(lambda s, bps: render_craftax_pixels_symbolic(s, bps, False), static_argnums=(1,))
            block_pixel_size = CRAFTAX_BLOCK_PIXEL_SIZE_HUMAN
        elif config["ENV_NAME"] == "Fabrax-Symbolic-v1":
            renderer = jax.jit(render_fabrax_pixels, static_argnums=(1,))
            block_pixel_size = FABRAX_BLOCK_PIXEL_SIZE_HUMAN

        @jax.jit
        def step_once(rng, obs, env_state):
            rng, rng_pi, rng_env = jax.random.split(rng, 3)
            player_skill = map_player_state_to_skill(env_state.player_state)
            pi, _ = network.apply(train_state["params"], obs, player_skill)
            action = pi.sample(seed=rng_pi)
            new_obs, new_env_state, reward, done, info = env_batch.step(rng_env, env_state, action, env_params)
            return rng, new_obs, new_env_state, reward, action

        # Reset and stream frames
        rng, _rng = jax.random.split(rng)
        obs, env_state = env_batch.reset(_rng, env_params)

        frames_list, states_list, rewards_list, actions_list = [], [], [], []
        print(f"Rendering env index: {render_env_idx}")
        for t in range(int(max_num_frames)):
            rng, obs, env_state, reward, action = step_once(rng, obs, env_state)

            # Slice the chosen env from the batch
            env_state_i = jax.tree.map(lambda x: x[render_env_idx], env_state)
            reward_i = float(reward[render_env_idx])
            state_i = int(env_state.player_state[render_env_idx])
            action_i = int(action[render_env_idx]) if hasattr(action, 'shape') else int(action)

            # Render and overlay text
            frame = renderer(env_state_i, block_pixel_size)
            frame_rgb = np.array(frame, dtype=np.float32)
            scale = max(1, 64 // int(block_pixel_size))
            if scale > 1:
                frame_rgb = np.repeat(np.repeat(frame_rgb, scale, axis=0), scale, axis=1)
            frame_with_text = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            cv2.putText(frame_with_text, f"Reward: {reward_i:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame_with_text, f"State: {state_i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame_with_text, f"Step: {t}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            frames_list.append(frame_with_text)
            states_list.append(state_i)
            rewards_list.append(reward_i)
            actions_list.append(action_i)

        frames = np.array(frames_list)
        states = np.array(states_list)
        actions = np.array(actions_list)
        rewards = np.array(rewards_list)
        # env_states not needed for simple path; return a placeholder (None)
        return frames, states, None, actions, rewards

    # Find a successful trajectory
    successful_trajectory = None
    max_attempts = 20
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} to find successful trajectory...")
        
        rng, _rng = jax.random.split(rng)
        obs, env_state = env_batch.reset(_rng, env_params)
        initial_carry = (rng, obs, env_state)
        
        inputs = jnp.arange(max_num_frames)
        _, (env_states, actions, rewards, rngs) = run_batch_rollout(initial_carry, inputs)
        
        # Check which environments achieved the goal
        max_states_per_env = jnp.max(env_states.player_state, axis=0)  # Max state per environment
        successful_envs = jnp.where(max_states_per_env >= goal_state)[0]
        
        if len(successful_envs) > 0:
            # Pick the first successful environment
            success_env_idx = successful_envs[0]
            print(f"SUCCESS: Environment {success_env_idx} achieved goal state {goal_state}")
            
            # Extract the successful trajectory
            def extract_env(x):
                return x[:, success_env_idx] if x.ndim > 1 else x[success_env_idx]
            
            successful_env_states = jax.tree.map(extract_env, env_states)
            successful_actions = actions[:, success_env_idx]
            successful_rewards = rewards[:, success_env_idx]  # Extract rewards too
            
            successful_trajectory = {
                'env_states': successful_env_states,
                'actions': successful_actions,
                'rewards': successful_rewards,  # Include rewards
            }
            break
        else:
            max_achieved = jnp.max(max_states_per_env)
            print(f"  No success. Best state achieved: {max_achieved}")
    
    if successful_trajectory is None:
        print(f"Failed to find successful trajectory after {max_attempts} attempts")
        return jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([])
    
    # Render the successful trajectory with reward information
    print("Rendering successful trajectory frames with rewards...")

    # Select appropriate renderer based on environment
    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        renderer = jax.jit(render_craftax_pixels, static_argnums=(1,))
        block_pixel_size = BLOCK_PIXEL_SIZE_HUMAN
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        renderer = jax.jit(lambda s, bps: render_craftax_pixels_symbolic(s, bps, False), static_argnums=(1,))
        block_pixel_size = CRAFTAX_BLOCK_PIXEL_SIZE_HUMAN
    elif config["ENV_NAME"] == "Fabrax-Symbolic-v1":
        renderer = jax.jit(render_fabrax_pixels, static_argnums=(1,))
        block_pixel_size = FABRAX_BLOCK_PIXEL_SIZE_HUMAN
    
    # Render each environment state directly
    def render_state_with_reward(env_state, reward, state, step):
        frame = renderer(env_state, block_pixel_size)
        # Upscale to ~64px tiles
        scale = max(1, 64 // int(block_pixel_size))
        frame_rgb = np.array(frame, dtype=np.float32)
        if scale > 1:
            frame_rgb = np.repeat(np.repeat(frame_rgb, scale, axis=0), scale, axis=1)
        # Convert to uint8 (renderer outputs 0..255 floats)
        frame_with_text = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        
        # Add reward text (top left)
        cv2.putText(
            frame_with_text,
            f"Reward: {float(reward):.3f}",
            (10, 30),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Font scale
            (255, 255, 255),  # Color (white)
            2,  # Thickness
        )
        
        # Add state text (top left, below reward)
        cv2.putText(
            frame_with_text,
            f"State: {int(state)}",
            (10, 60),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Font scale
            (255, 255, 255),  # Color (white)
            2,  # Thickness
        )
        
        # Add step text (top left, below state)
        cv2.putText(
            frame_with_text,
            f"Step: {int(step)}",
            (10, 90),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            0.8,  # Font scale
            (255, 255, 255),  # Color (white)
            2,  # Thickness
        )
        
        return frame_with_text
    
    # Render all frames with reward information
    env_states_trajectory = successful_trajectory['env_states']
    rewards_trajectory = successful_trajectory['rewards']
    num_frames = len(env_states_trajectory.player_state)
    
    print(f"Rendering {num_frames} frames with reward information")
    
    frames_list = []
    for i in range(num_frames):
        # Extract single frame data
        env_state_i = jax.tree.map(lambda x: x[i], env_states_trajectory)
        reward_i = rewards_trajectory[i]
        state_i = env_states_trajectory.player_state[i]
        
        # Render frame with reward info
        frame_with_info = render_state_with_reward(env_state_i, reward_i, state_i, i)
        frames_list.append(frame_with_info)
        
        if i % 50 == 0:
            print(f"  Rendered frame {i}/{num_frames}")
    
    frames = np.array(frames_list)
    
    # Extract states, actions, and rewards
    states = successful_trajectory['env_states'].player_state
    env_states = successful_trajectory['env_states'] 
    actions = successful_trajectory['actions']
    rewards = successful_trajectory['rewards']
    
    print(f"Generated {len(frames)} frames for successful trajectory")
    print(f"Total reward: {float(jnp.sum(rewards)):.3f}")
    print(f"Average reward per step: {float(jnp.mean(rewards)):.3f}")
    
    return frames, states, env_states, actions, rewards


def render_video_with_rewards(frames, states, rewards, video_path):
    """
    Render video with reward information already embedded in frames.
    """
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

    for frame in frames:
        # Convert to BGR for saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release everything when the job is finished
    out.release()


def generate_ppo_videos_with_rewards(
    policy_path: str,
    output_dir: str = "videos_with_rewards",
    max_frames: int = 2000,
    num_videos: int = 3,
    env_name: str = None
):
    """
    Generate multiple example videos with reward information from PPO policies.
    Supports both Craftax-Classic and Fabrax environments.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating videos with rewards for PPO policy: {policy_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of videos: {num_videos}")

    for video_idx in range(num_videos):
        try:
            print(f"\n=== Generating video {video_idx + 1}/{num_videos} ===")

            # Generate frames with reward information
            frames, states, env_states, actions, rewards = gen_frames_ppo_with_rewards(
                policy_path=policy_path,
                max_num_frames=max_frames,
                num_envs=64,
                env_name=env_name
            )

            if len(frames) == 0:
                print(f"    Failed to generate trajectory for video {video_idx + 1}")
                continue

            # Create video filename
            policy_name = os.path.basename(policy_path.rstrip('/'))
            video_filename = f"{policy_name}_ppo_rewards_video{video_idx + 1}.mp4"
            video_path = os.path.join(output_dir, video_filename)

            # Render video with embedded reward information
            render_video_with_rewards(frames, states, rewards, video_path)

            print(f"    ✅ Saved video: {video_path}")
            print(f"       Frames: {len(frames)}")
            print(f"       Total reward: {float(jnp.sum(rewards)):.3f}")
            print(f"       Avg reward/step: {float(jnp.mean(rewards)):.3f}")

            # Also save reward data as text file
            reward_filename = f"{policy_name}_ppo_rewards_data{video_idx + 1}.txt"
            reward_path = os.path.join(output_dir, reward_filename)

            with open(reward_path, 'w') as f:
                f.write(f"Reward data for {video_filename}\n")
                f.write(f"PPO Policy: {policy_path}\n")
                f.write(f"Total frames: {len(frames)}\n")
                f.write(f"Total reward: {float(jnp.sum(rewards)):.3f}\n")
                f.write(f"Average reward per step: {float(jnp.mean(rewards)):.3f}\n\n")
                f.write("Step\tReward\n")
                for i, reward in enumerate(rewards):
                    f.write(f"{i}\t{float(reward):.6f}\n")

            print(f"    📊 Saved reward data: {reward_path}")

        except Exception as e:
            print(f"    ❌ Error generating video {video_idx + 1}: {e}")
            continue

    print(f"\n🎬 PPO video generation with rewards completed! Check {output_dir}/ for videos and data.")


def generate_ppo_rnn_videos_with_rewards(
    policy_path: str,
    output_dir: str = "videos_with_rewards",
    max_frames: int = 2000,
    num_videos: int = 3,
    env_name: str = None
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating videos with rewards for PPO-RNN policy: {policy_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of videos: {num_videos}")

    for video_idx in range(num_videos):
        try:
            print(f"\n=== Generating RNN video {video_idx + 1}/{num_videos} ===")
            frames, states, env_states, actions, rewards = gen_frames_ppo_rnn_with_rewards(
                policy_path=policy_path,
                max_num_frames=max_frames,
                num_envs=64,
                env_name=env_name,
            )
            if len(frames) == 0:
                print(f"    Failed to generate RNN trajectory for video {video_idx + 1}")
                continue
            policy_name = os.path.basename(policy_path.rstrip('/'))
            video_filename = f"{policy_name}_ppo_rnn_rewards_video{video_idx + 1}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            render_video_with_rewards(frames, states, rewards, video_path)
            print(f"    ✅ Saved video: {video_path}")
        except Exception as e:
            print(f"    ❌ Error generating RNN video {video_idx + 1}: {e}")
            continue
    print(f"\n🎬 PPO-RNN video generation completed! Check {output_dir}/ for videos.")


def generate_policy_videos_with_rewards(
    policy_path: str,
    output_dir: str = "videos_with_rewards",
    max_frames: int = 2000,
    goal_states: list = None,
    num_videos: int = 3,
    require_goal_state = True
):
    """
    Generate multiple example videos with reward information displayed.
    Supports both Craftax-Classic and Fabrax environments (hierarchical policies).
    """
    if goal_states is None:
        goal_states = [5, 8, 11]  # Different difficulty levels
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating videos with rewards for policy: {policy_path}")
    print(f"Output directory: {output_dir}")
    print(f"Goal states: {goal_states}")
    print(f"Videos per goal state: {num_videos}")
    
    for goal_state in goal_states:
        print(f"\n=== Generating videos for goal state {goal_state} ===")
        
        for video_idx in range(num_videos):
            try:
                print(f"  Generating video {video_idx + 1}/{num_videos}...")
                
                # Generate frames with reward information
                frames, states, env_states, actions, rewards = gen_frames_hierarchical_with_rewards(
                    policy_path=policy_path,
                    max_num_frames=max_frames,
                    goal_state=goal_state,
                    num_envs=3,
                    require_goal_state=require_goal_state

                )
                
                if len(frames) == 0:
                    print(f"    Failed to generate trajectory for goal state {goal_state}")
                    continue
                
                # Create video filename
                policy_name = os.path.basename(policy_path.rstrip('/'))
                video_filename = f"{policy_name}_goal{goal_state}_rewards_video{video_idx + 1}.mp4"
                video_path = os.path.join(output_dir, video_filename)
                
                # Render video with embedded reward information
                render_video_with_rewards(frames, states, rewards, video_path)
                
                print(f"    ✅ Saved video: {video_path}")
                print(f"       Frames: {len(frames)}, Max state: {max(states)}")
                print(f"       Total reward: {float(jnp.sum(rewards)):.3f}")
                print(f"       Avg reward/step: {float(jnp.mean(rewards)):.3f}")
                
                # Also save reward data as text file
                reward_filename = f"{policy_name}_goal{goal_state}_rewards_data{video_idx + 1}.txt"
                reward_path = os.path.join(output_dir, reward_filename)
                
                with open(reward_path, 'w') as f:
                    f.write(f"Reward data for {video_filename}\n")
                    f.write(f"Goal state: {goal_state}\n")
                    f.write(f"Total frames: {len(frames)}\n")
                    f.write(f"Total reward: {float(jnp.sum(rewards)):.3f}\n")
                    f.write(f"Average reward per step: {float(jnp.mean(rewards)):.3f}\n\n")
                    f.write("Step\tState\tReward\n")
                    for i, (state, reward) in enumerate(zip(states, rewards)):
                        f.write(f"{i}\t{int(state)}\t{float(reward):.6f}\n")
                
                print(f"    📊 Saved reward data: {reward_path}")
                
            except Exception as e:
                print(f"    ❌ Error generating video {video_idx + 1}: {e}")
                continue
    
    print(f"\n🎬 Video generation with rewards completed! Check {output_dir}/ for videos and data.")


def test_video_generation_with_rewards():
    """Test video generation with rewards using the saved policy."""
    policy_path = "/home/renos/flow-rl/exp/ppo_mlp_fabrax"

    print("=== Testing PPO Video Generation with Rewards ===")

    generate_ppo_videos_with_rewards(
        policy_path=policy_path,
        output_dir="test_ppo_videos_rewards",
        max_frames=1500,  # Shorter for testing
        num_videos=1,  # Just 1 video for testing
        env_name="Fabrax-Symbolic-v1"  # Specify the environment name for testing
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate videos with reward information for Craftax-Classic or Fabrax environments")
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to trained policy directory (e.g., /path/to/exp/ppo_mlp_fabrax)")
    parser.add_argument("--output_dir", type=str, default="videos_with_rewards",
                       help="Directory to save videos")
    parser.add_argument("--max_frames", type=int, default=2000,
                       help="Maximum frames per video")
    parser.add_argument("--goal_states", type=int, nargs="+", default=[5, 8, 11],
                       help="Goal states to attempt (for hierarchical policies)")
    parser.add_argument("--num_videos", type=int, default=3,
                       help="Number of videos per goal state (hierarchical) or total videos (PPO)")
    parser.add_argument("--ppo", action="store_true",
                       help="Use PPO (non-hierarchical) policy instead of hierarchical flow policy")
    parser.add_argument("--ppo_rnn", action="store_true",
                       help="Use PPO-RNN policy video generation")
    parser.add_argument("--env_name", type=str, default=None,
                       help="Environment name to use (e.g., Craftax-Classic-Symbolic-v1, Fabrax-Symbolic-v1)")
    parser.add_argument("--test", action="store_true",
                       help="Run test with predefined settings")
    parser.add_argument("--simple", action="store_true", help="if true don't require the target state to be reached for a video to be generated")
    
    args = parser.parse_args()
    
    if args.test:
        test_video_generation_with_rewards()
    elif args.ppo:
        generate_ppo_videos_with_rewards(
            policy_path=args.policy_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            num_videos=args.num_videos,
            env_name=args.env_name
        )
    elif args.ppo_rnn:
        generate_ppo_rnn_videos_with_rewards(
            policy_path=args.policy_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            num_videos=args.num_videos,
            env_name=args.env_name,
        )
    else:
        generate_policy_videos_with_rewards(
            policy_path=args.policy_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            goal_states=args.goal_states,
            num_videos=args.num_videos,
            require_goal_state = not args.simple
        )
