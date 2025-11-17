from flowrl.utils.test import load_policy_params
import jax

jax.config.update("jax_compilation_cache_dir", "/home/renos/jax_cache/")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_flow_env_from_name
import yaml
import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from .logz.batch_logging import batch_log, create_log_dict
from .wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
import importlib

# Import TransformerXL components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../transformerXL_PPO_JAX'))
from transformerXL import Transformer
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl
# TransformerXL integration from transformerXL_PPO_JAX


class ActorCriticTransformer(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    num_tasks: int  # For Flow environment task indexing
    gating: bool = False
    gating_bias: float = 0.

    def setup(self):
        # Separate transformer backbone for each task
        self.transformers = [
            Transformer(
                encoder_size=self.encoder_size,
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                num_layers=self.num_layers,
                gating=self.gating,
                gating_bias=self.gating_bias
            )
            for _ in range(self.num_tasks)
        ]

        # Initialize separate actor and critic networks for each task
        self.actor_networks = [
            self.create_actor_network()
            for _ in range(self.num_tasks)
        ]
        self.critic_networks = [
            self.create_critic_network()
            for _ in range(self.num_tasks)
        ]

    def create_actor_network(self):
        """Create a 3-layer MLP for actor"""
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        layers = []
        # Hidden layer 1
        layers.append(
            nn.Dense(
                self.hidden_layers,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        )
        layers.append(activation)
        # Hidden layer 2
        layers.append(
            nn.Dense(
                self.hidden_layers,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        )
        layers.append(activation)
        # Output layer
        layers.append(
            nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )
        )
        return nn.Sequential(layers)

    def create_critic_network(self):
        """Create a 3-layer MLP for critic"""
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        layers = []
        # Hidden layer 1
        layers.append(
            nn.Dense(
                self.hidden_layers,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        )
        layers.append(activation)
        # Hidden layer 2
        layers.append(
            nn.Dense(
                self.hidden_layers,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        )
        layers.append(activation)
        # Output layer
        layers.append(
            nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )
        )
        return nn.Sequential(layers)

    def __call__(self, memories, obs, state_skills, mask):
        """Standard forward (used when not specifically needing eval or train mode)"""
        # Process through all task-specific transformers
        transformer_outputs = []
        memory_outputs = []
        for i in range(self.num_tasks):
            x, mem_out = self.transformers[i].forward_eval(memories, obs, mask)
            transformer_outputs.append(x)
            memory_outputs.append(mem_out)

        # Stack transformer outputs: [batch_size, num_tasks, encoder_size]
        x_all = jnp.stack(transformer_outputs, axis=1)
        memory_out_all = jnp.stack(memory_outputs, axis=1)

        # Select transformer output based on state_skills
        task_ids_one_hot = jnp.eye(self.num_tasks)[state_skills]  # Shape: [batch_size, num_tasks]
        x = jnp.sum(x_all * task_ids_one_hot[:, :, None], axis=1)  # [batch_size, encoder_size]
        memory_out = jnp.sum(memory_out_all * task_ids_one_hot[:, :, None, None], axis=1)  # [batch_size, num_layers, encoder_size]

        # Process input through all actor and critic networks
        actor_outputs = jnp.stack(
            [jax.vmap(self.actor_networks[i])(x) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size, num_tasks, action_dim]

        critic_outputs = jnp.stack(
            [jax.vmap(self.critic_networks[i])(x) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size, num_tasks, 1]

        # Select the correct actor and critic output based on state_skills
        selected_actor_outputs = jnp.sum(
            actor_outputs * task_ids_one_hot[:, :, None], axis=1
        )
        selected_critic_outputs = jnp.sum(
            critic_outputs * task_ids_one_hot[:, :, None], axis=1
        ).squeeze(-1)

        pi = distrax.Categorical(logits=selected_actor_outputs)
        return pi, selected_critic_outputs, memory_out

    def model_forward_eval(self, memories, obs, state_skills, mask):
        """Used during environment rollout (single timestep of obs). Returns the memory"""
        # Process through all task-specific transformers
        transformer_outputs = []
        memory_outputs = []
        for i in range(self.num_tasks):
            x, mem_out = self.transformers[i].forward_eval(memories, obs, mask)
            transformer_outputs.append(x)
            memory_outputs.append(mem_out)

        # Stack transformer outputs: [batch_size, num_tasks, encoder_size]
        x_all = jnp.stack(transformer_outputs, axis=1)
        memory_out_all = jnp.stack(memory_outputs, axis=1)

        # Select transformer output based on state_skills
        task_ids_one_hot = jnp.eye(self.num_tasks)[state_skills]  # Shape: [batch_size, num_tasks]
        x = jnp.sum(x_all * task_ids_one_hot[:, :, None], axis=1)  # [batch_size, encoder_size]
        memory_out = jnp.sum(memory_out_all * task_ids_one_hot[:, :, None, None], axis=1)  # [batch_size, num_layers, encoder_size]

        # Process input through all actor and critic networks
        actor_outputs = jnp.stack(
            [jax.vmap(self.actor_networks[i])(x) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size, num_tasks, action_dim]

        critic_outputs = jnp.stack(
            [jax.vmap(self.critic_networks[i])(x) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size, num_tasks, 1]

        # Select the correct actor and critic output based on state_skills
        task_ids_one_hot = jnp.eye(self.num_tasks)[state_skills]  # Shape: [batch_size, num_tasks]

        # Use one-hot encoding to select the correct outputs
        selected_actor_outputs = jnp.sum(
            actor_outputs * task_ids_one_hot[:, :, None], axis=1
        )
        selected_critic_outputs = jnp.sum(
            critic_outputs * task_ids_one_hot[:, :, None], axis=1
        ).squeeze(-1)

        pi = distrax.Categorical(logits=selected_actor_outputs)
        return pi, selected_critic_outputs, memory_out

    def model_forward_train(self, memories, obs, state_skills, mask):
        """Used during training: a window of observation is sent. Don't return the memory"""
        # Process through all task-specific transformers
        transformer_outputs = []
        for i in range(self.num_tasks):
            x_task = self.transformers[i].forward_train(memories, obs, mask)
            transformer_outputs.append(x_task)

        # Stack transformer outputs: [batch_size, num_tasks, window_grad, encoder_size]
        x_all = jnp.stack(transformer_outputs, axis=1)
        batch_size, num_tasks, window_size, features = x_all.shape

        # Reshape state_skills to [batch_size, 1, window_size] for broadcasting
        state_skills_expanded = state_skills[:, None, :]  # [batch_size, 1, window_size]
        task_ids_one_hot = jnp.eye(self.num_tasks)[state_skills_expanded]  # [batch_size, 1, window_size, num_tasks]
        task_ids_one_hot = task_ids_one_hot.squeeze(1)  # [batch_size, window_size, num_tasks]

        # Select transformer output: [batch_size, window_size, encoder_size]
        x = jnp.einsum('btnf,btn->btf', x_all.transpose(0, 2, 1, 3), task_ids_one_hot)

        # Process input through all actor and critic networks
        # x shape: [batch_size, window_grad, encoder_size]

        # Reshape to [batch_size * window_size, features] for processing
        x_flat = x.reshape(-1, features)

        actor_outputs = jnp.stack(
            [jax.vmap(self.actor_networks[i])(x_flat) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size*window_size, num_tasks, action_dim]

        critic_outputs = jnp.stack(
            [jax.vmap(self.critic_networks[i])(x_flat) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size*window_size, num_tasks, 1]

        # Reshape state_skills to match
        state_skills_flat = state_skills.reshape(-1)  # [batch_size * window_size]

        # Select the correct actor and critic output based on state_skills
        task_ids_one_hot = jnp.eye(self.num_tasks)[state_skills_flat]  # Shape: [batch_size*window_size, num_tasks]

        # Use one-hot encoding to select the correct outputs
        selected_actor_outputs = jnp.sum(
            actor_outputs * task_ids_one_hot[:, :, None], axis=1
        )
        selected_critic_outputs = jnp.sum(
            critic_outputs * task_ids_one_hot[:, :, None], axis=1
        ).squeeze(-1)

        # Reshape back to [batch_size, window_size, ...]
        selected_actor_outputs = selected_actor_outputs.reshape(batch_size, window_size, -1)
        selected_critic_outputs = selected_critic_outputs.reshape(batch_size, window_size)

        pi = distrax.Categorical(logits=selected_actor_outputs)
        return pi, selected_critic_outputs


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    player_state: jnp.ndarray
    inventory: jnp.ndarray


# Helper functions for memory and batch operations
indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def make_train(config, prev_model_state=None, return_test_network=False):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if config["MODULE_PATH"]:
        module_path = config["MODULE_PATH"]
        module_name = "reward_and_state"
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        # Step 3: Create a new module based on the spec
        module = importlib.util.module_from_spec(spec)

        # Step 4: Execute the module in its own namespace
        # Note: This actually runs the module code and initializes any variables/functions
        spec.loader.exec_module(module)
        # module = importlib.import_module(module_name)
        module_dict = module.__dict__
    else:
        module_dict = None
    env = make_craftax_flow_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"], module_dict
    )
    env_params = env.default_params

    # Update env_params with intrinsics_reward setting
    intrinsics_reward = config.get("INTRINSICS_REWARD", False)
    if hasattr(env_params, "intrinsics_reward"):
        env_params = env_params.replace(intrinsics_reward=intrinsics_reward)

    heads, num_heads = env.heads_info

    heads = list(heads)
    task_to_skill_index = jnp.array(env.task_to_skill_index)
    num_tasks_ = len(task_to_skill_index)

    def map_player_state_to_skill(player_state):
        # Create a one-hot encoding of task_to_skill_index
        player_state_one_hot = jnp.eye(num_tasks_)[player_state]
        player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)

        return player_skill

    num_tasks = env.num_tasks

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    if prev_model_state is not None:
        prev_state = prev_model_state
    else:
        prev_state = {}

    def param_updater(new_state):
        if "params" not in prev_state:
            return new_state
        for key, value in prev_state["params"]["params"].items():
            new_state["params"][key] = value
        return new_state

    if return_test_network:
        network = ActorCriticTransformer(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            hidden_layers=config["HIDDEN_LAYERS"],
            encoder_size=config["EMBED_SIZE"],
            num_heads=config["NUM_HEADS"],
            qkv_features=config["QKV_FEATURES"],
            num_layers=config["NUM_LAYERS"],
            num_tasks=num_heads,
            gating=config.get("GATING", False),
            gating_bias=config.get("GATING_BIAS", 0.),
        )
        return network

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCriticTransformer(
                action_dim=env.action_space(env_params).n,
                activation=config["ACTIVATION"],
                hidden_layers=config["HIDDEN_LAYERS"],
                encoder_size=config["EMBED_SIZE"],
                num_heads=config["NUM_HEADS"],
                qkv_features=config["QKV_FEATURES"],
                num_layers=config["NUM_LAYERS"],
                num_tasks=num_heads,
                gating=config.get("GATING", False),
                gating_bias=config.get("GATING_BIAS", 0.),
            )
        else:
            assert 0, "Not implemented atm - only symbolic environments supported"

        # Check if we're doing continuation (full TrainState with optimizer) vs new expert (just params)
        is_continuation = prev_model_state is not None and hasattr(prev_model_state, 'opt_state')

        if is_continuation:
            # Continuation: use the restored TrainState directly to preserve optimizer state
            train_state = prev_model_state
        else:
            # New expert or fresh training: create new TrainState with fresh optimizer
            rng, _rng = jax.random.split(rng)
            init_obs = jnp.zeros((2, *env.observation_space(env_params).shape))
            init_memory = jnp.zeros((2, config["WINDOW_MEM"], config["NUM_LAYERS"], config["EMBED_SIZE"]))
            init_mask = jnp.zeros((2, config["NUM_HEADS"], 1, config["WINDOW_MEM"] + 1), dtype=jnp.bool_)
            init_states = jnp.zeros((2,), dtype=jnp.int32)
            network_params = network.init(_rng, init_memory, init_obs, init_states, init_mask)
            network_params = jax.pure_callback(
                param_updater, network_params, network_params
            )
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

        ex_state = {}

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # INITIALIZE MEMORIES AND MASKS
        memories = jnp.zeros((config["NUM_ENVS"], config["WINDOW_MEM"], config["NUM_LAYERS"], config["EMBED_SIZE"]))
        memories_mask = jnp.zeros((config["NUM_ENVS"], config["NUM_HEADS"], 1, config["WINDOW_MEM"] + 1), dtype=jnp.bool_)
        # memories_mask_idx starts at WINDOW_MEM+1 so that first decrement brings it to WINDOW_MEM
        memories_mask_idx = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32) + (config["WINDOW_MEM"] + 1)
        done = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_)

        # Initialize latest-state snapshot tracking per-env
        ex_state = {
            **ex_state,
            "best_player_state": env_state.env_state.player_state,
            "latest_obs": obsv,
            "latest_inner_state": env_state.env_state,
            "latest_memories": memories,
            "latest_memories_mask": memories_mask,
            "latest_memories_mask_idx": memories_mask_idx,
        }

        # Initialize progressive curriculum state
        if config.get("PROGRESSIVE_RESET_CURRICULUM", False):
            ex_state["progressive_reset_state"] = 0

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, step_env_currentloop):
                (
                    train_state,
                    env_state,
                    memories,
                    memories_mask,
                    memories_mask_idx,
                    last_obs,
                    done,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # Reset memories mask and mask idx in case of done
                memories_mask_idx = jnp.where(
                    done,
                    config["WINDOW_MEM"],
                    jnp.clip(memories_mask_idx - 1, 0, config["WINDOW_MEM"])
                )
                memories_mask = jnp.where(
                    done[:, None, None, None],
                    jnp.zeros((config["NUM_ENVS"], config["NUM_HEADS"], 1, config["WINDOW_MEM"] + 1), dtype=jnp.bool_),
                    memories_mask
                )

                # Update memories mask with the potential additional step taken into account at this step
                memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config["WINDOW_MEM"] + 1)
                memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config["NUM_HEADS"], 1)
                memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

                # SELECT ACTION
                player_state = env_state.env_state.player_state
                state_skill = map_player_state_to_skill(player_state)
                inv = env_state.env_state.inventory

                rng, _rng = jax.random.split(rng)
                pi, value, memories_out = network.apply(
                    train_state.params, memories, last_obs, state_skill, memories_mask,
                    method=network.model_forward_eval
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # ADD THE CACHED ACTIVATIONS IN MEMORIES FOR NEXT STEP
                memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done_raw, info = env.step(
                    _rng, env_state, action, env_params
                )
                done = info["task_done"]

                # Helper for batched masking
                def apply_mask(mask, new, old):
                    m = mask
                    while m.ndim < new.ndim:
                        m = m[..., None]
                    return jnp.where(m, new, old)

                # Update latest snapshot when player_state strictly increases
                new_ps = env_state.env_state.player_state
                prev_best = ex_state["best_player_state"]

                # For progressive curriculum, only track snapshots up to current frontier
                if config.get("PROGRESSIVE_RESET_CURRICULUM", False):
                    current_reset_state = ex_state.get("progressive_reset_state", 0)
                    is_relevant = new_ps <= current_reset_state
                    improved = jnp.logical_and(new_ps > prev_best, is_relevant)
                else:
                    improved = new_ps > prev_best

                latest_obs = apply_mask(improved, obsv, ex_state["latest_obs"])
                latest_inner = jax.tree.map(
                    lambda n, o: apply_mask(improved, n, o),
                    env_state.env_state,
                    ex_state["latest_inner_state"],
                )
                latest_memories = apply_mask(
                    improved, memories, ex_state["latest_memories"]
                )
                latest_memories_mask = apply_mask(
                    improved, memories_mask, ex_state["latest_memories_mask"]
                )
                latest_memories_mask_idx = jnp.where(
                    improved,
                    memories_mask_idx,
                    ex_state["latest_memories_mask_idx"],
                )
                best_ps = jnp.where(improved, new_ps, prev_best)
                ex_state = {
                    **ex_state,
                    "best_player_state": best_ps,
                    "latest_obs": latest_obs,
                    "latest_inner_state": latest_inner,
                    "latest_memories": latest_memories,
                    "latest_memories_mask": latest_memories_mask,
                    "latest_memories_mask_idx": latest_memories_mask_idx,
                }

                # Latest-state reset override at episode boundary (post AutoReset):
                # Trigger only on the single transition into 0
                was_zero_reset = jnp.logical_and(done, new_ps == 0)

                # Track actual transition outcomes BEFORE optimistic reset
                # These are needed for success/failure metrics later
                actual_success_transition = jnp.logical_and(
                    done, new_ps == (player_state + 1)
                )
                actual_failure_transition = jnp.logical_and(
                    done, new_ps == 0
                )
                info["was_success_transition"] = actual_success_transition
                info["was_failure_transition"] = actual_failure_transition

                # Optionally only enable latest-reset on specific skill(s)
                if config.get("PROGRESSIVE_RESET_CURRICULUM", False):
                    # Progressive curriculum: goal is advancing from current_reset_state to current_reset_state+1
                    # This prevents bias from only resetting failures
                    current_reset_state = ex_state.get("progressive_reset_state", 0)
                    was_goal_state = jnp.logical_and(done, new_ps == current_reset_state + 1)
                    zero_or_goal = jnp.logical_or(was_zero_reset, was_goal_state)
                    # Reset environments at or past the current frontier
                    at_or_past_current = player_state >= current_reset_state
                    reset_gate = jnp.logical_and(zero_or_goal, at_or_past_current)
                elif config.get("LATEST_RESET_FRONTIER_ONLY", False):
                    # Original: only reset at frontier (last) skill
                    was_goal_state = jnp.logical_and(done, new_ps == config["SUCCESS_STATE_INDEX"])
                    zero_or_goal = jnp.logical_or(was_zero_reset, was_goal_state)
                    frontier_state = config["SUCCESS_STATE_INDEX"] - 1
                    on_frontier = (player_state == frontier_state)
                    reset_gate = jnp.logical_and(zero_or_goal, on_frontier)
                else:
                    was_goal_state = jnp.logical_and(done, new_ps == config["SUCCESS_STATE_INDEX"])
                    zero_or_goal = jnp.logical_or(was_zero_reset, was_goal_state)
                    reset_gate = zero_or_goal

                if config.get("LATEST_RESET_PROB", 0.0) > 0:
                    rng, rng_choice = jax.random.split(rng)
                    choose_latest = jax.random.bernoulli(
                        rng_choice, p=config["LATEST_RESET_PROB"], shape=reset_gate.shape
                    )
                    use_latest = jnp.logical_and(reset_gate, choose_latest)
                else:
                    use_latest = jnp.zeros_like(reset_gate, dtype=jnp.bool_)
                keep_reset = jnp.logical_and(reset_gate, jnp.logical_not(use_latest))

                memories_mask_transition = memories_mask.squeeze()

                # Apply chosen next state/obs and restore TransformerXL memories
                env_inner_selected = jax.tree.map(
                    lambda latest, cur: apply_mask(use_latest, latest, cur),
                    ex_state["latest_inner_state"],
                    env_state.env_state,
                )
                env_state = env_state.replace(env_state=env_inner_selected)
                obsv = apply_mask(use_latest, ex_state["latest_obs"], obsv)
                memories = apply_mask(
                    use_latest, ex_state["latest_memories"], memories
                )
                memories_mask = apply_mask(
                    use_latest, ex_state["latest_memories_mask"], memories_mask
                )
                memories_mask_idx = jnp.where(
                    use_latest,
                    ex_state["latest_memories_mask_idx"],
                    memories_mask_idx,
                )

                # Optionally rebase snapshot to the new 0 state when we keep the reset
                if config.get("LATEST_RESET_REBASE_ON_FRESH", False):
                    reset_memories = jnp.zeros_like(ex_state["latest_memories"])
                    reset_memories_mask = jnp.zeros_like(
                        ex_state["latest_memories_mask"]
                    )
                    reset_memories_idx = (
                        jnp.zeros_like(ex_state["latest_memories_mask_idx"])
                        + (config["WINDOW_MEM"] + 1)
                    )
                    ex_state = {
                        **ex_state,
                        "best_player_state": apply_mask(
                            keep_reset, new_ps, ex_state["best_player_state"]
                        ),
                        "latest_obs": apply_mask(
                            keep_reset, obsv, ex_state["latest_obs"]
                        ),
                        "latest_inner_state": jax.tree.map(
                            lambda new, old: apply_mask(keep_reset, new, old),
                            env_state.env_state,
                            ex_state["latest_inner_state"],
                        ),
                        "latest_memories": apply_mask(
                            keep_reset, reset_memories, ex_state["latest_memories"]
                        ),
                        "latest_memories_mask": apply_mask(
                            keep_reset,
                            reset_memories_mask,
                            ex_state["latest_memories_mask"],
                        ),
                        "latest_memories_mask_idx": jnp.where(
                            keep_reset,
                            reset_memories_idx,
                            ex_state["latest_memories_mask_idx"],
                        ),
                    }

                done_carry = jnp.where(use_latest, jnp.zeros_like(done), done)

                # COMPUTE THE INDICES OF THE FINAL MEMORIES THAT ARE TAKEN INTO ACCOUNT IN THIS STEP
                # Not forgetting that we will concatenate the previous WINDOW_MEM to the NUM_STEPS
                memory_indices = jnp.arange(0, config["WINDOW_MEM"])[None, :] + step_env_currentloop * jnp.ones((config["NUM_ENVS"], 1), dtype=jnp.int32)

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    memories_mask=memories_mask_transition,
                    memories_indices=memory_indices,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                    player_state=player_state,
                    inventory=inv,
                )
                runner_state = (
                    train_state,
                    env_state,
                    memories,
                    memories_mask,
                    memories_mask_idx,
                    obsv,
                    done_carry,  # Clear memories on actual resets; preserve latest reuse
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, (transition, memories_out)

            # Copy the first memories before the new rollout to concatenate with new steps
            memories_previous = runner_state[2]

            # SCAN THE STEP TO GET THE TRANSITIONS AND CACHED MEMORIES
            runner_state, (traj_batch, memories_batch) = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                memories,
                memories_mask,
                memories_mask_idx,
                last_obs,
                done,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            state_skill = map_player_state_to_skill(env_state.env_state.player_state)
            _, last_val, _ = network.apply(
                train_state.params, memories, last_obs, state_skill, memories_mask,
                method=network.model_forward_eval
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ADD PREVIOUS WINDOW_MEM TO THE CURRENT NUM_STEPS SO THAT FIRST STEPS USE MEMORIES FROM PREVIOUS
            memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(update_state_mb, batch_info):
                    train_state, agg = update_state_mb
                    traj_batch, memories_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, memories_batch, gae, targets):
                        # USE THE CACHED MEMORIES ONLY FROM THE FIRST STEP OF A WINDOW GRAD
                        # Construct the memory batch from memory indices
                        memories_batch_selected = batch_indices_select(
                            memories_batch, traj_batch.memories_indices[:, ::config["WINDOW_GRAD"]]
                        )
                        memories_batch_selected = batchify(memories_batch_selected)

                        # CREATE THE MASK FOR WINDOW GRAD
                        memories_mask = traj_batch.memories_mask.reshape(
                            (-1, config["WINDOW_GRAD"],) + traj_batch.memories_mask.shape[2:]
                        )
                        memories_mask = jnp.swapaxes(memories_mask, 1, 2)
                        # Concatenate with 0s to fill before the roll
                        memories_mask = jnp.concatenate(
                            (memories_mask, jnp.zeros(memories_mask.shape[:-1] + (config["WINDOW_GRAD"] - 1,), dtype=jnp.bool_)),
                            axis=-1
                        )
                        # Roll of different value for each step to match the right
                        memories_mask = roll_vmap(memories_mask, jnp.arange(0, config["WINDOW_GRAD"]), -1)

                        # RESHAPE observations and trajectories for windowed training
                        obs = traj_batch.obs
                        obs = obs.reshape((-1, config["WINDOW_GRAD"],) + obs.shape[2:])

                        # Reshape player_state for task-specific head selection
                        player_state = traj_batch.player_state.reshape((-1, config["WINDOW_GRAD"]))
                        state_skill = map_player_state_to_skill(player_state)

                        traj_batch_reshaped, targets_reshaped, gae_reshaped = jax.tree.map(
                            lambda x: jnp.reshape(x, (-1, config["WINDOW_GRAD"]) + x.shape[2:]),
                            (traj_batch, targets, gae)
                        )

                        # NETWORK OUTPUT
                        pi, value = network.apply(
                            params, memories_batch_selected, obs, state_skill, memories_mask,
                            method=network.model_forward_train
                        )

                        log_prob = pi.log_prob(traj_batch_reshaped.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch_reshaped.value + (
                            value - traj_batch_reshaped.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets_reshaped)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_reshaped)
                        value_loss_per_sample = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch_reshaped.log_prob)
                        gae_normalized = (gae_reshaped - gae_reshaped.mean()) / (gae_reshaped.std() + 1e-8)
                        loss_actor1 = ratio * gae_normalized
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_normalized
                        )
                        loss_actor_per_sample = -jnp.minimum(loss_actor1, loss_actor2)

                        # Per-skill dynamic balancing (weights per sample)
                        if config.get("PER_SKILL_BALANCE", True):
                            # state_skill has shape [-1, WINDOW_GRAD]
                            skill_ids_flat = state_skill.reshape(-1)
                            num_samples = skill_ids_flat.shape[0]
                            counts = jnp.bincount(skill_ids_flat, length=num_heads).astype(jnp.int32)
                            threshold = jnp.int32(config.get("PER_SKILL_BALANCE_THRESHOLD", 64))
                            cap = jnp.float32(config.get("PER_SKILL_BALANCE_CAP", 4.0))
                            counts_f = jnp.maximum(counts.astype(jnp.float32), 1.0)
                            raw_scale = jnp.clip(num_samples / counts_f, 1.0, cap)
                            enable = (counts >= threshold).astype(jnp.float32)
                            per_skill_scale = 1.0 + (raw_scale - 1.0) * enable
                            weights = per_skill_scale[skill_ids_flat].reshape(state_skill.shape)
                        else:
                            weights = jnp.ones_like(loss_actor_per_sample, dtype=jnp.float32)

                        weight_sum = jnp.maximum(weights.sum(), 1.0)
                        # Weighted means for actor/value losses
                        loss_actor = (weights * loss_actor_per_sample).sum() / weight_sum
                        value_loss = (weights * value_loss_per_sample).sum() / weight_sum
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        # Per-skill diagnostics (computed on raw, unnormalized advantages)
                        # Reshape back to compute per-skill stats on the full trajectory
                        skill_ids = state_skill.reshape(-1)
                        counts = jnp.bincount(skill_ids, length=num_heads).astype(jnp.float32)

                        # Approx KL using action log-probs
                        approx_kl = (traj_batch_reshaped.log_prob - log_prob).reshape(-1)
                        entropies = pi.entropy().reshape(-1)
                        vloss_per_sample_flat = value_loss_per_sample.reshape(-1)
                        gae_raw_flat = gae_reshaped.reshape(-1)

                        adv_sum = jnp.bincount(skill_ids, weights=gae_raw_flat, length=num_heads)
                        adv_sq_sum = jnp.bincount(skill_ids, weights=gae_raw_flat * gae_raw_flat, length=num_heads)
                        kl_sum = jnp.bincount(skill_ids, weights=approx_kl, length=num_heads)
                        ent_sum = jnp.bincount(skill_ids, weights=entropies, length=num_heads)
                        vloss_sum = jnp.bincount(
                            skill_ids, weights=vloss_per_sample_flat, length=num_heads
                        )

                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            kl_sum,
                            ent_sum,
                            vloss_sum,
                            adv_sum,
                            adv_sq_sum,
                            counts,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (out, grads) = grad_fn(
                        train_state.params, traj_batch, memories_batch, advantages, targets
                    )
                    total_loss, (
                        _vl,
                        _la,
                        _ent,
                        kl_sum,
                        ent_sum,
                        vloss_sum,
                        adv_sum,
                        adv_sq_sum,
                        counts,
                    ) = out
                    train_state = train_state.apply_gradients(grads=grads)

                    # Update aggregation
                    agg = {
                        "kl_sum": agg["kl_sum"] + kl_sum,
                        "ent_sum": agg["ent_sum"] + ent_sum,
                        "vloss_sum": agg["vloss_sum"] + vloss_sum,
                        "adv_sum": agg["adv_sum"] + adv_sum,
                        "adv_sq_sum": agg["adv_sq_sum"] + adv_sq_sum,
                        "count_sum": agg["count_sum"] + counts,
                    }

                    losses = (total_loss, 0)
                    return (train_state, agg), losses

                (
                    train_state,
                    traj_batch,
                    memories_batch,
                    advantages,
                    targets,
                    rng,
                    agg,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # Ensure NUM_STEPS is divisible by WINDOW_GRAD for proper batching
                assert (
                    config["NUM_STEPS"] % config["WINDOW_GRAD"] == 0
                ), "NUM_STEPS should be divisible by WINDOW_GRAD to properly batch the window_grad"

                # PERMUTE ALONG THE NUM_ENVS ONLY NOT TO LOSE TRACK FROM TEMPORAL
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (traj_batch, memories_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1),
                    batch,
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                (train_state, agg), losses = jax.lax.scan(
                    _update_minbatch, (train_state, agg), minibatches
                )

                update_state = (
                    train_state,
                    traj_batch,
                    memories_batch,
                    advantages,
                    targets,
                    rng,
                    agg,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                memories_batch,
                advantages,
                targets,
                rng,
                {
                    "kl_sum": jnp.zeros((num_heads,), dtype=jnp.float32),
                    "ent_sum": jnp.zeros((num_heads,), dtype=jnp.float32),
                    "vloss_sum": jnp.zeros((num_heads,), dtype=jnp.float32),
                    "adv_sum": jnp.zeros((num_heads,), dtype=jnp.float32),
                    "adv_sq_sum": jnp.zeros((num_heads,), dtype=jnp.float32),
                    "count_sum": jnp.zeros((num_heads,), dtype=jnp.float32),
                },
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            agg = update_state[6]

            # Per-skill diagnostics (averaged over all epochs and minibatches)
            counts = jnp.maximum(agg["count_sum"], 1.0)
            per_skill_kl = agg["kl_sum"] / counts
            per_skill_entropy = agg["ent_sum"] / counts
            per_skill_value_loss = agg["vloss_sum"] / counts
            adv_mean = agg["adv_sum"] / counts
            adv_var = jnp.maximum(agg["adv_sq_sum"] / counts - adv_mean * adv_mean, 1e-8)
            adv_std = jnp.sqrt(adv_var)

            # Per-skill reward tracking
            skill_ids_full = map_player_state_to_skill(traj_batch.player_state)
            rewards = traj_batch.reward.astype(jnp.float32)
            skill_done = traj_batch.info["task_done"].astype(jnp.float32)

            def _skill_reward_scan(carry, inputs):
                running_sum = carry + inputs[0]
                done_step = inputs[1]
                completed_reward = running_sum * done_step
                running_sum = running_sum * (1.0 - done_step)
                return running_sum, completed_reward

            init_running = ex_state.get(
                "skill_reward_running",
                jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32),
            )
            final_running, completed_rewards = jax.lax.scan(
                _skill_reward_scan,
                init_running,
                (rewards, skill_done),
            )
            ex_state = {**ex_state, "skill_reward_running": final_running}

            skill_ids_flat = skill_ids_full.reshape(-1)
            completed_rewards_flat = completed_rewards.reshape(-1)
            skill_done_flat = skill_done.reshape(-1)
            skill_episode_reward_sum = jnp.bincount(
                skill_ids_flat,
                weights=completed_rewards_flat,
                length=num_heads,
            )
            skill_episode_counts = jnp.bincount(
                skill_ids_flat,
                weights=skill_done_flat,
                length=num_heads,
            )
            per_skill_reward = jnp.where(
                skill_episode_counts > 0,
                skill_episode_reward_sum / skill_episode_counts,
                0.0,
            )

            def process(x, returned_episode, is_special=False, is_nearest=False):
                # reached_ep = returned_episode
                reached_ep = traj_batch.player_state == config["SUCCESS_STATE_INDEX"]
                if is_nearest:
                    # mask_seen = (x[:, :, :, 0, :] < 30) and (x[:, :, :, 1, :] < 30)
                    mask_seen = jnp.logical_and(
                        x[:, :, :, 0, :] < 30, x[:, :, :, 1, :] < 30
                    )
                    return (mask_seen * returned_episode[:, :, None, None]).sum(
                        axis=(0, 1)
                    ) / returned_episode.sum()
                elif is_special:
                    # Reshape returned_episode to match the dimensions of x for broadcasting
                    # also we want returned episode to track the success rates directly
                    return (x * returned_episode[:, :, None]).sum(
                        axis=(0, 1)
                    ) / returned_episode.sum()
                else:
                    return (x * returned_episode).sum() / returned_episode.sum()

            metric = {
                key: process(
                    value,
                    traj_batch.info["returned_episode"],
                    is_special=(key == "reached_state"),
                    is_nearest=(key == "closest_blocks"),
                )
                for key, value in traj_batch.info.items()
            }
            player_state_one_hot = jax.nn.one_hot(
                traj_batch.player_state, num_classes=18
            )
            state_occurrences = player_state_one_hot.sum(
                axis=(0, 1)
            )  # Sum over batch and timesteps
            total_weights = traj_batch.info["returned_episode"].sum()
            state_rates = state_occurrences / total_weights
            metric["state_rates"] = state_rates

            # Attach per-skill diagnostics to metric
            metric["per_skill_kl"] = per_skill_kl
            metric["per_skill_entropy"] = per_skill_entropy
            metric["per_skill_value_loss"] = per_skill_value_loss
            metric["per_skill_reward"] = per_skill_reward
            metric["per_skill_adv_mean"] = adv_mean
            metric["per_skill_adv_std"] = adv_std
            metric["per_skill_counts"] = counts
            metric["per_skill_episode_counts"] = skill_episode_counts

            # Calculate per-skill transition success/failure from done episodes
            # Use the actual transition outcomes saved in info BEFORE optimistic resets
            # This prevents optimistic resets from hiding true success/failure outcomes
            player_state_before = traj_batch.player_state
            was_success_transition = traj_batch.info["was_success_transition"]
            was_failure_transition = traj_batch.info["was_failure_transition"]

            # Aggregate per skill
            skill_ids_full = map_player_state_to_skill(player_state_before)
            skill_ids_flat = skill_ids_full.reshape(-1)
            success_transitions_flat = was_success_transition.reshape(-1).astype(jnp.float32)
            failure_transitions_flat = was_failure_transition.reshape(-1).astype(jnp.float32)

            per_skill_successes = jnp.bincount(
                skill_ids_flat,
                weights=success_transitions_flat,
                length=num_heads,
            )
            per_skill_failures = jnp.bincount(
                skill_ids_flat,
                weights=failure_transitions_flat,
                length=num_heads,
            )

            # Calculate current batch success rate (raw, no prior smoothing)
            # successes / (successes + failures), with epsilon to avoid division by zero
            # EMA will handle smoothing over time
            per_skill_total_transitions = per_skill_successes + per_skill_failures
            batch_success_rate = per_skill_successes / (per_skill_total_transitions + 1e-8)

            # Exponential moving average to track recent performance (decay old failures)
            # EMA update: new = alpha * batch + (1-alpha) * old
            # Higher alpha (e.g., 0.2) = faster adaptation, lower = more smoothing
            ema_alpha = config.get("SUCCESS_RATE_EMA_ALPHA", 0.1)
            prev_ema = ex_state.get(
                "per_skill_success_rate_ema",
                jnp.zeros((num_heads,), dtype=jnp.float32),  # Initialize to 0
            )
            per_skill_success_rate_ema = ema_alpha * batch_success_rate + (1.0 - ema_alpha) * prev_ema
            ex_state = {**ex_state, "per_skill_success_rate_ema": per_skill_success_rate_ema}

            # Progressive curriculum: advance to next state when current state is mastered
            if config.get("PROGRESSIVE_RESET_CURRICULUM", False):
                current_reset_state = ex_state.get("progressive_reset_state", 0)
                threshold = config.get("PROGRESSIVE_RESET_THRESHOLD", 0.2)
                frontier = config["SUCCESS_STATE_INDEX"] - 1

                # Check if current state has exceeded threshold and we're not at frontier yet
                # Map state to skill index to get the right success rate
                current_skill = map_player_state_to_skill(jnp.array([current_reset_state]))[0]
                current_success_rate = per_skill_success_rate_ema[current_skill]

                can_advance = jnp.logical_and(
                    current_success_rate >= threshold,
                    current_reset_state < frontier
                )

                # Advance to next state if threshold met
                new_reset_state = jnp.where(
                    can_advance,
                    current_reset_state + 1,
                    current_reset_state
                )
                ex_state = {**ex_state, "progressive_reset_state": new_reset_state}
                metric["progressive_reset_state"] = new_reset_state
                metric["progressive_threshold_met"] = can_advance.astype(jnp.float32)
            else:
                metric["progressive_reset_state"] = -1  # Not using progressive mode

            metric["per_skill_success_rate"] = per_skill_success_rate_ema
            metric["per_skill_success_rate_batch"] = batch_success_rate
            metric["per_skill_successes"] = per_skill_successes
            metric["per_skill_failures"] = per_skill_failures
            metric["per_skill_total_transitions"] = per_skill_total_transitions

            # Calculate transition success rate: b/(a+b) where
            # a = transitions from (goal_state-1) to 0 (failure)
            # b = transitions from (goal_state-1) to goal_state (success)
            goal_state = config["SUCCESS_STATE_INDEX"]
            penultimate_state = goal_state - 1

            # Find transitions from penultimate state
            current_states = traj_batch.player_state[:-1]  # All states except last
            next_states = traj_batch.player_state[1:]      # All states except first
            episode_mask = traj_batch.info["returned_episode"][1:]  # Only count completed episodes

            # Count transitions from penultimate_state to 0 (failure)
            failure_transitions = jnp.logical_and(
                jnp.logical_and(current_states == penultimate_state, next_states == 0),
                episode_mask > 0  # Only count in completed episodes
            ).sum()

            # Count transitions from penultimate_state to goal_state (success)
            success_transitions = jnp.logical_and(
                jnp.logical_and(current_states == penultimate_state, next_states == goal_state),
                episode_mask > 0  # Only count in completed episodes
            ).sum()

            # Calculate success rate: b/(a+b), handle division by zero
            total_transitions = failure_transitions + success_transitions
            transition_success_rate = jnp.where(
                total_transitions > 0,
                success_transitions / total_transitions,
                0.0
            )

            metric["transition_success_rate"] = transition_success_rate
            metric["failure_transitions"] = failure_transitions
            metric["success_transitions"] = success_transitions

            # inventory when done
            # Handle inventory differently for Craftax Classic vs full Craftax vs Fabrax
            if "Classic" in config["ENV_NAME"] or "Fabrax" in config["ENV_NAME"]:
                # Craftax Classic & Fabrax: all inventory fields are scalars, can use tree_flatten
                flat_values, _ = jax.tree_util.tree_flatten(traj_batch.inventory)
                inventory = jnp.stack(flat_values, axis=0).astype(jnp.float16)
            else:
                # Full Craftax: has array fields (armour, potions), handle separately
                scalar_inventory = jnp.stack([
                    traj_batch.inventory.wood,
                    traj_batch.inventory.stone,
                    traj_batch.inventory.coal,
                    traj_batch.inventory.iron,
                    traj_batch.inventory.diamond,
                    traj_batch.inventory.sapling,
                    traj_batch.inventory.pickaxe,
                    traj_batch.inventory.sword,
                    traj_batch.inventory.bow,
                    traj_batch.inventory.arrows,
                    traj_batch.inventory.torches,
                    traj_batch.inventory.ruby,
                    traj_batch.inventory.sapphire,
                    traj_batch.inventory.books,
                ], axis=0).astype(jnp.float16)

                # Stack array fields along the first axis to match scalar_inventory shape
                armour_stacked = jnp.moveaxis(traj_batch.inventory.armour, -1, 0).astype(jnp.float16)
                potions_stacked = jnp.moveaxis(traj_batch.inventory.potions, -1, 0).astype(jnp.float16)
                inventory = jnp.concatenate([scalar_inventory, armour_stacked, potions_stacked], axis=0)
            # done_point = traj_batch.done
            done_point = traj_batch.player_state == config["SUCCESS_STATE_INDEX"]
            avg_item = (
                jax.nn.one_hot(inventory, num_classes=10) * done_point[None, ..., None]
            ).sum(axis=(-3, -2)) / done_point.sum()
            metric["average_item"] = avg_item

            # Extract rng from update_state (now at index 5, since agg is at -1)
            rng = update_state[5]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_state,
                env_state,
                memories,
                memories_mask,
                memories_mask_idx,
                last_obs,
                done,  # This is the done from the last env step, will be used to clear memories
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            memories,
            memories_mask,
            memories_mask_idx,
            obsv,
            done,  # Initially False, will become keep_reset in subsequent iterations
            ex_state,
            _rng,
            0,
        )

        # While loop for success state progression
        def cond_fun(loop_carry):
            i, _, success_state_rate, _ = loop_carry
            jax.debug.print("success_state_rate {x}", x=success_state_rate)
            return jnp.logical_and(
                (i < config["NUM_UPDATES"] - 1),
                (success_state_rate < config["SUCCESS_STATE_RATE"]),
            )

        # Define the body function
        def body_fun(loop_carry):
            i, runner_state, _, metric = loop_carry
            runner_state, metric = _update_step(runner_state, None)
            success_state_rate = metric["reached_state"][config["SUCCESS_STATE_INDEX"]]
            return i + 1, runner_state, success_state_rate, metric

        # Execute the while loop
        runner_state, metric = _update_step(runner_state, None)
        # Initialize the loop carry
        initial_carry = (0, runner_state, 0, metric)
        num_steps, runner_state, success_rate, metric = jax.lax.while_loop(
            cond_fun, body_fun, initial_carry
        )
        return {
            "runner_state": runner_state,
            "num_steps": num_steps,
            "info": metric,
        }

    return train


def run_ppo(config, training_state_i=None):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    if training_state_i is None:
        training_state_i = config["SUCCESS_STATE"]

    if config["USE_WANDB"]:
        # Create unique run name including training phase info
        training_phase = "initial" if config.get("SUCCESS_STATE_RATE", 0) < 0.1 else "final"
        run_name = (
            config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M"
            + f"-state{training_state_i}"
            + f"-{training_phase}"
            + "-txrl"
        )

        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=run_name,
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])
    config["SUCCESS_STATE_INDEX"] = training_state_i

    def restore_model(path):
        config_path = f"{path}/config.yaml"
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = {
            k.upper(): v["value"] if type(v) == dict and "value" in v else v
            for k, v in config.items()
        }
        train_state = load_policy_params(path)

        return train_state

    if "PREV_MODULE_PATH" in config and config["PREV_MODULE_PATH"]:
        directory, file_name = os.path.split(config["PREV_MODULE_PATH"])
        # Remove the .py extension from the file name
        file_name_without_extension = os.path.splitext(file_name)[0]
        new_directory = os.path.join(
            directory, f"{file_name_without_extension}_policies"
        )
        train_state = restore_model(new_directory)
    else:
        train_state = None

    train_jit = jax.jit(make_train(config, prev_model_state=train_state))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    jax.block_until_ready(out)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    success_rate = float(out["info"]["reached_state"][0][training_state_i])

    # Compute mean episode length over completed episodes and frames-per-success
    mel = out["info"]["returned_episode_lengths"]
    # Handle vmapped dimension
    if hasattr(mel, "shape") and len(mel.shape) > 0:
        mean_episode_length = float(mel[0])
    else:
        mean_episode_length = float(mel)
    frames_per_success = float(mean_episode_length / success_rate) if success_rate > 0 else None

    # Derive how many updates and timesteps we actually trained
    try:
        trained_updates = int(np.array(out["num_steps"][0]))
    except Exception:
        try:
            trained_updates = int(np.array(out["num_steps"]))
        except Exception:
            trained_updates = None
    if trained_updates is not None:
        trained_timesteps = int(trained_updates * config["NUM_STEPS"] * config["NUM_ENVS"])
        print(f"Trained updates: {trained_updates}, trained timesteps: {trained_timesteps}")
        # Stash into config so it is saved alongside the checkpoint
        config["TRAINED_UPDATES"] = trained_updates
        config["TRAINED_TIMESTEPS"] = trained_timesteps

    # Persist episode metrics for downstream scheduling/LLM heuristics
    if mean_episode_length is not None:
        config["MEAN_EPISODE_LENGTH"] = mean_episode_length
    if frames_per_success is not None:
        config["FRAMES_PER_SUCCESS"] = frames_per_success
    # Persist overall success rate for downstream prompts/metrics
    config["SUCCESS_RATE"] = success_rate

    # Finish wandb run to separate from next training phase
    if config["USE_WANDB"]:
        wandb.finish()

    def _save_network(rs_index, dir_name):
        train_states = out["runner_state"][rs_index]
        train_state = jax.tree.map(lambda x: x[0], train_states)
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)

        # dir_name already contains the full path including "policies"
        checkpoint_manager = CheckpointManager(dir_name, orbax_checkpointer, options)
        print(f"saved runner state to {dir_name}")
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            int(config["TOTAL_TIMESTEPS"]),
            train_state,
            save_kwargs={"save_args": save_args},
        )

    if config["MODULE_PATH"]:
        directory, file_name = os.path.split(config["MODULE_PATH"])
        # Remove the .py extension from the file name
        file_name_without_extension = os.path.splitext(file_name)[0]
        new_directory = os.path.join(
            directory, f"{file_name_without_extension}_policies"
        )
        os.makedirs(new_directory, exist_ok=True)
        _save_network(0, new_directory + "/policies")
        # save config
        config_path = f"{new_directory}/config.yaml"
        print(f"Saving config to: {config_path}")
        print(f"Config contains {len(config)} keys: {list(config.keys())[:10]}...")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            f.flush()
            os.fsync(f.fileno())
        print(f"Config saved successfully to: {config_path}")
    else:
        assert 0, "should be in module path"

    return out, success_rate > config["SUCCESS_STATE_RATE"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e7
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--latest_reset_prob", type=float, default=0.0,
                        help="Probability of resetting to the latest saved state on termination instead of the start state.")
    parser.add_argument("--latest_reset_frontier_only", action=argparse.BooleanOptionalAction, default=False,
                        help="If true, apply latest-state reset only when on the frontier (last) skill being trained.")
    parser.add_argument("--latest_reset_rebase_on_fresh", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, when choosing a fresh reset (1-p), also rebase the saved snapshot to the new 0 state.")

    # Progressive Curriculum Params
    parser.add_argument("--progressive_reset_curriculum", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable progressive curriculum learning that starts at state 0 and advances when success rate exceeds threshold.")
    parser.add_argument("--progressive_reset_threshold", type=float, default=0.2,
                        help="Success rate threshold (EMA) for advancing to next state in progressive curriculum.")
    parser.add_argument("--success_rate_ema_alpha", type=float, default=0.15,
                        help="EMA smoothing factor for per-skill success rate (higher = faster adaptation).")

    # TransformerXL Params
    parser.add_argument("--embed_size", type=int, default=256, help="Transformer embedding dimension")
    parser.add_argument("--hidden_layers", type=int, default=256, help="Hidden layer size for actor/critic heads")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--qkv_features", type=int, default=256, help="QKV feature dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--window_mem", type=int, default=16, help="Memory window size")
    parser.add_argument("--window_grad", type=int, default=8, help="Gradient window size for training")
    parser.add_argument("--gating", action="store_true", help="Use gating in transformer")
    parser.add_argument("--gating_bias", type=float, default=0.0, help="Gating bias value")

    # Flow Params
    parser.add_argument(
        "--module_path", type=str, default="/home/renos/flow-rl/exp/premade/1.py"
    )
    parser.add_argument(
        "--prev_module_path", type=str, default=None,
        help="Optional: path to previous module .py to restore seed policies from (<stem>_policies/policies/)."
    )
    # what success rate to achieve before optimizing next node
    parser.add_argument("--success_state_rate", type=float, default=0.8)

    parser.add_argument("--success_state", type=int)
    parser.add_argument("--intrinsics-reward", action="store_true", help="Add intrinsics (hunger/thirst/fatigue) to reward")

    # Per-skill dynamic balancing (effective per-skill LR)
    parser.add_argument("--per_skill_balance", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable dynamic per-skill loss balancing based on minibatch share.")
    parser.add_argument("--per_skill_balance_threshold", type=int, default=64,
                        help="Minimum per-skill samples in minibatch required to apply balancing to that skill.")
    parser.add_argument("--per_skill_balance_cap", type=float, default=4.0,
                        help="Maximum upweighting factor for any skill (caps num_samples/count).")

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
