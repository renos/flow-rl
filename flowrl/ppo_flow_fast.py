from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Any, Dict, List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import wandb
from flax.core import frozen_dict
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from craftax.craftax_env import make_craftax_flow_env_from_name

from flowrl.utils.test import load_policy_params
from .logz.batch_logging import batch_log, create_log_dict
from .models.actor_critic import ActorCriticMoE
from .wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from jax import tree_util

jax.config.update("jax_compilation_cache_dir", "/home/renos/jax_cache/")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    player_state: jnp.ndarray
    inventory: Any


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Any
    last_obs: jnp.ndarray
    rng: jax.Array


def _import_optional_module(module_path: str | None) -> Dict[str, Any] | None:
    if not module_path:
        return None
    module_name = "reward_and_state"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def make_train_fast(
    config: Dict[str, Any],
    prev_model_state: TrainState | None = None,
    return_test_network: bool = False,
):
    config = dict(config)
    config.setdefault("NUM_STEPS", 64)
    config.setdefault("NUM_ENVS", 1024)
    config.setdefault("NUM_MINIBATCHES", 8)
    config.setdefault("TOTAL_TIMESTEPS", int(1e7))
    config.setdefault("UPDATE_EPOCHS", 4)
    config.setdefault("LR", 2e-4)
    config.setdefault("ANNEAL_LR", True)
    config.setdefault("USE_OPTIMISTIC_RESETS", True)
    config.setdefault("MAX_GRAD_NORM", 1.0)
    config.setdefault("CLIP_EPS", 0.2)
    config.setdefault("ENT_COEF", 0.01)
    config.setdefault("VF_COEF", 0.5)
    config.setdefault("GAMMA", 0.99)
    config.setdefault("GAE_LAMBDA", 0.95)
    config.setdefault("DEBUG", True)
    config.setdefault("USE_WANDB", True)
    config.setdefault("SUCCESS_STATE_RATE", 0.8)
    config.setdefault("LAYER_SIZE", 512)
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    module_dict = _import_optional_module(config.get("MODULE_PATH"))

    env = make_craftax_flow_env_from_name(
        config["ENV_NAME"], not config.get("USE_OPTIMISTIC_RESETS", True), module_dict
    )
    env_params = env.default_params

    intrinsics_reward = config.get("INTRINSICS_REWARD", False)
    if hasattr(env_params, "intrinsics_reward"):
        env_params = env_params.replace(intrinsics_reward=intrinsics_reward)

    heads, num_heads = env.heads_info
    task_to_skill_index = jnp.array(env.task_to_skill_index)
    config["NUM_HEADS"] = num_heads
    num_tasks = int(task_to_skill_index.shape[0])

    def map_player_state_to_skill(player_state):
        return jnp.take(task_to_skill_index, player_state, axis=0).astype(jnp.int32)

    env = LogWrapper(env)
    if config.get("USE_OPTIMISTIC_RESETS", True):
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config.get("OPTIMISTIC_RESET_RATIO", 16), config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    config_frozen = frozen_dict.freeze(config)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config_frozen["NUM_MINIBATCHES"] * config_frozen["UPDATE_EPOCHS"]))
            / config_frozen["NUM_UPDATES"]
        )
        return config_frozen["LR"] * frac

    prev_state = prev_model_state if prev_model_state is not None else {}

    def param_updater(new_state):
        if "params" not in prev_state:
            return new_state
        for key, value in prev_state["params"]["params"].items():
            new_state["params"][key] = value
        return new_state

    if return_test_network:
        return ActorCriticMoE(
            action_dim=env.action_space(env_params).n,
            layer_width=config_frozen["LAYER_SIZE"],
            num_layers=4,
            num_tasks=num_heads,
        )

    def compute_metrics_host(traj_batch):
        traj_np = jax.tree_map(lambda x: np.array(x), traj_batch)
        info = traj_np.info if hasattr(traj_np, "info") else {}
        returned_episode = info.get("returned_episode")
        if returned_episode is None:
            returned_episode = np.ones_like(traj_np.reward)
        weight = float(returned_episode.sum())
        denom = weight if weight > 0 else 1.0

        metrics: Dict[str, Any] = {}

        reached_state = info.get("reached_state")
        if reached_state is not None:
            metrics["reached_state"] = (
                reached_state * returned_episode[:, :, None]
            ).sum(axis=(0, 1)) / denom
        else:
            metrics["reached_state"] = np.zeros(config_frozen["NUM_HEADS"], dtype=np.float32)

        closest_blocks = info.get("closest_blocks")
        if closest_blocks is not None:
            mask_seen = np.logical_and(
                closest_blocks[:, :, :, 0, :] < 30,
                closest_blocks[:, :, :, 1, :] < 30,
            )
            metrics["closest_blocks"] = (
                mask_seen * returned_episode[:, :, None, None]
            ).sum(axis=(0, 1)) / denom

        player_state_one_hot = np.eye(max(num_tasks, 1), dtype=np.float32)[
            np.array(traj_np.player_state)
        ]
        state_occurrences = player_state_one_hot.sum(axis=(0, 1))
        metrics["state_rates"] = state_occurrences / denom

        goal_state = int(config_frozen["SUCCESS_STATE_INDEX"])
        penultimate_state = max(goal_state - 1, 0)
        if traj_np.player_state.shape[0] > 1:
            current_states = traj_np.player_state[:-1]
            next_states = traj_np.player_state[1:]
            episode_mask = returned_episode[1:]
            failure_transitions = np.logical_and(
                np.logical_and(current_states == penultimate_state, next_states == 0),
                episode_mask > 0,
            ).sum()
            success_transitions = np.logical_and(
                np.logical_and(current_states == penultimate_state, next_states == goal_state),
                episode_mask > 0,
            ).sum()
            total_transitions = failure_transitions + success_transitions
            metrics["failure_transitions"] = float(failure_transitions)
            metrics["success_transitions"] = float(success_transitions)
            metrics["transition_success_rate"] = (
                float(success_transitions) / float(total_transitions)
                if total_transitions > 0
                else 0.0
            )
        else:
            metrics["failure_transitions"] = 0.0
            metrics["success_transitions"] = 0.0
            metrics["transition_success_rate"] = 0.0

        if "Classic" in config_frozen["ENV_NAME"] or "Fabrax" in config_frozen["ENV_NAME"]:
            flat_values, _ = tree_util.tree_flatten(traj_batch.inventory)
            inventory = np.stack([np.array(v) for v in flat_values], axis=0).astype(np.int32)
        else:
            scalar_inventory = np.stack(
                [
                    np.array(traj_batch.inventory.wood),
                    np.array(traj_batch.inventory.stone),
                    np.array(traj_batch.inventory.coal),
                    np.array(traj_batch.inventory.iron),
                    np.array(traj_batch.inventory.diamond),
                    np.array(traj_batch.inventory.sapling),
                    np.array(traj_batch.inventory.pickaxe),
                    np.array(traj_batch.inventory.sword),
                    np.array(traj_batch.inventory.bow),
                    np.array(traj_batch.inventory.arrows),
                    np.array(traj_batch.inventory.torches),
                    np.array(traj_batch.inventory.ruby),
                    np.array(traj_batch.inventory.sapphire),
                    np.array(traj_batch.inventory.books),
                ],
                axis=0,
                dtype=np.int32,
            )
            armour_stacked = np.moveaxis(
                np.array(traj_batch.inventory.armour), -1, 0
            ).astype(np.int32)
            potions_stacked = np.moveaxis(
                np.array(traj_batch.inventory.potions), -1, 0
            ).astype(np.int32)
            inventory = np.concatenate([scalar_inventory, armour_stacked, potions_stacked], axis=0)

        done_point = (np.array(traj_np.player_state) == goal_state).astype(np.float32)
        done_total = float(done_point.sum())
        if done_total > 0:
            avg_item = (
                np.eye(10, dtype=np.float32)[inventory] * done_point[None, ..., None]
            ).sum(axis=(-3, -2)) / done_total
        else:
            avg_item = np.zeros((inventory.shape[0], 10), dtype=np.float32)
        metrics["average_item"] = avg_item

        return metrics

    def train(rng):
        if "Symbolic" not in config_frozen["ENV_NAME"]:
            raise NotImplementedError("Conv policy not yet supported in fast variant")

        network = ActorCriticMoE(
            action_dim=env.action_space(env_params).n,
            layer_width=config_frozen["LAYER_SIZE"],
            num_layers=4,
            num_tasks=num_heads,
        )

        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape), dtype=jnp.float32)
        init_states = jnp.zeros((1,), dtype=jnp.int32)
        network_params = network.init(init_rng, init_x, init_states)
        network_params = jax.pure_callback(param_updater, network_params, network_params)

        if config_frozen["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config_frozen["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config_frozen["MAX_GRAD_NORM"]),
                optax.adam(config_frozen["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, reset_rng = jax.random.split(rng)
        last_obs, env_state = env.reset(reset_rng, env_params)
        runner_state = RunnerState(train_state, env_state, last_obs, rng)

        @jax.jit
        def rollout_once(train_state, env_state, last_obs, rng):
            def _env_step(carry, _):
                env_state, obs, rng = carry
                player_state = env_state.env_state.player_state
                inventory = env_state.env_state.inventory
                rng, action_rng, env_rng = jax.random.split(rng, 3)
                state_skill = map_player_state_to_skill(player_state)
                pi, value = network.apply(train_state.params, obs, state_skill)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)
                next_obs, next_env_state, reward, done, info = env.step(
                    env_rng, env_state, action, env_params
                )
                done = info["task_done"]
                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=obs,
                    next_obs=next_obs,
                    info=info,
                    player_state=player_state,
                    inventory=inventory,
                )
                return (next_env_state, next_obs, rng), transition

            init_carry = (env_state, last_obs, rng)
            (new_env_state, new_obs, new_rng), traj_batch = jax.lax.scan(
                _env_step,
                init_carry,
                None,
                length=config_frozen["NUM_STEPS"],
                unroll=16,
            )
            return new_env_state, new_obs, new_rng, traj_batch

        @jax.jit
        def compute_gae(train_state, env_state, last_obs, traj_batch):
            state_skill = map_player_state_to_skill(env_state.env_state.player_state)
            _, last_val = network.apply(train_state.params, last_obs, state_skill)

            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done = transition.done
                value = transition.value
                reward = transition.reward
                delta = reward + config_frozen["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config_frozen["GAMMA"]
                    * config_frozen["GAE_LAMBDA"]
                    * (1 - done)
                    * gae
                )
                return (gae, value), gae

            (_, _), advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            targets = advantages + traj_batch.value
            return advantages, targets

        @jax.jit
        def ppo_update(train_state, traj_batch, advantages, targets, rng):
            batch_size = (
                config_frozen["MINIBATCH_SIZE"] * config_frozen["NUM_MINIBATCHES"]
            )
            batch = (traj_batch, advantages, targets)

            def _reshape(x):
                return x.reshape((batch_size,) + x.shape[2:])

            flat_batch = jax.tree_map(_reshape, batch)
            loss_init = jnp.zeros(4, dtype=jnp.float32)

            def _update_minibatch(carry, minibatch):
                train_state, loss_acc = carry
                traj_mb, adv_mb, target_mb = minibatch

                def _loss_fn(params):
                    state_skill = map_player_state_to_skill(traj_mb.player_state)
                    pi, value = network.apply(params, traj_mb.obs, state_skill)
                    log_prob = pi.log_prob(traj_mb.action)
                    value_pred_clipped = traj_mb.value + (
                        value - traj_mb.value
                    ).clip(-config_frozen["CLIP_EPS"], config_frozen["CLIP_EPS"])
                    value_losses = jnp.square(value - target_mb)
                    value_losses_clipped = jnp.square(value_pred_clipped - target_mb)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    ratio = jnp.exp(log_prob - traj_mb.log_prob)
                    gae = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config_frozen["CLIP_EPS"],
                            1.0 + config_frozen["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    total_loss = (
                        loss_actor
                        + config_frozen["VF_COEF"] * value_loss
                        - config_frozen["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                (total_loss, aux), grads = jax.value_and_grad(
                    _loss_fn, has_aux=True
                )(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                value_loss, loss_actor, entropy = aux
                loss_values = jnp.array(
                    [total_loss, value_loss, loss_actor, entropy], dtype=jnp.float32
                )
                loss_acc = loss_acc + loss_values
                return (train_state, loss_acc), None

            def _epoch_step(carry, _):
                train_state, rng, loss_acc = carry
                rng, perm_rng = jax.random.split(rng)
                permutation = jax.random.permutation(perm_rng, batch_size)
                shuffled = jax.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), flat_batch
                )
                minibatches = jax.tree_map(
                    lambda x: jnp.reshape(
                        x,
                        (
                            config_frozen["NUM_MINIBATCHES"],
                            config_frozen["MINIBATCH_SIZE"],
                        )
                        + x.shape[1:],
                    ),
                    shuffled,
                )
                (train_state, loss_acc), _ = jax.lax.scan(
                    _update_minibatch,
                    (train_state, loss_acc),
                    minibatches,
                    unroll=4,
                )
                return (train_state, rng, loss_acc), None

            (train_state, rng, loss_acc), _ = jax.lax.scan(
                _epoch_step,
                (train_state, rng, loss_init),
                None,
                length=config_frozen["UPDATE_EPOCHS"],
                unroll=4,
            )
            total_batches = (
                config_frozen["UPDATE_EPOCHS"] * config_frozen["NUM_MINIBATCHES"]
            )
            loss_mean = loss_acc / total_batches
            return train_state, rng, loss_mean

        #@jax.jit
        def update_step(runner_state):
            train_state, env_state, last_obs, rng = runner_state
            env_state, last_obs, rng, traj_batch = rollout_once(
                train_state, env_state, last_obs, rng
            )
            advantages, targets = compute_gae(
                train_state, env_state, last_obs, traj_batch
            )
            train_state, rng, loss_mean = ppo_update(
                train_state, traj_batch, advantages, targets, rng
            )
            return RunnerState(train_state, env_state, last_obs, rng), traj_batch, loss_mean

        metrics = {}
        loss_summary = np.zeros(4, dtype=np.float32)
        success_rate = 0.0
        updates_ran = 0

        for update_idx in range(int(config_frozen["NUM_UPDATES"])):
            print(update_idx)
            runner_state_local, traj_batch, loss_mean = update_step(runner_state)
            runner_state = runner_state_local
            metrics = compute_metrics_host(traj_batch)
            loss_summary = np.array(loss_mean)
            metrics["loss_total"] = float(loss_summary[0])
            metrics["loss_value"] = float(loss_summary[1])
            metrics["loss_actor"] = float(loss_summary[2])
            metrics["entropy"] = float(loss_summary[3])

            if bool(config_frozen["DEBUG"]) and bool(config_frozen["USE_WANDB"]):
                to_log = create_log_dict(metrics, config_frozen)
                batch_log(update_idx, to_log, config_frozen)

            reached_state = metrics.get("reached_state")
            if reached_state is not None and len(reached_state) > goal_state:
                success_rate = float(reached_state[goal_state])
            else:
                success_rate = 0.0

            updates_ran = update_idx + 1
            if success_rate >= config_frozen["SUCCESS_STATE_RATE"]:
                break

        return {
            "runner_state": runner_state,
            "num_steps": updates_ran,
            "info": metrics,
        }

    return train


def run_ppo_fast(config, training_state_i: int | None = None):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    if training_state_i is None:
        training_state_i = config["SUCCESS_STATE"]

    if config.get("USE_WANDB", False):
        training_phase = (
            "initial" if config.get("SUCCESS_STATE_RATE", 0) < 0.1 else "final"
        )
        run_name = (
            config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M"
            + f"-state{training_state_i}"
            + f"-{training_phase}"
        )
        wandb.init(
            project=config.get("WANDB_PROJECT"),
            entity=config.get("WANDB_ENTITY"),
            config=config,
            name=run_name,
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])
    config["SUCCESS_STATE_INDEX"] = training_state_i

    def restore_model(path):
        config_path = f"{path}/config.yaml"
        with open(config_path, "r") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader)
        loaded = {
            k.upper(): v["value"] if isinstance(v, dict) and "value" in v else v
            for k, v in loaded.items()
        }
        train_state = load_policy_params(path)
        return train_state

    if config.get("PREV_MODULE_PATH"):
        directory, file_name = os.path.split(config["PREV_MODULE_PATH"])
        file_name_without_extension = os.path.splitext(file_name)[0]
        new_directory = os.path.join(
            directory, f"{file_name_without_extension}_policies"
        )
        prev_state = restore_model(new_directory)
    else:
        prev_state = None

    train_fn = make_train_fast(config, prev_model_state=prev_state)

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for seed_idx, rng_key in enumerate(rngs):
        result = train_fn(rng_key)
        runner_state = result["runner_state"]
        jax.block_until_ready(runner_state.last_obs)
        results.append(result)
    elapsed = time.time() - t0
    print("Time to run experiment", elapsed)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / max(elapsed, 1e-6))

    success_metric = results[0]["info"].get("reached_state")
    if success_metric is not None and len(success_metric) > training_state_i:
        success_rate = float(success_metric[training_state_i])
    else:
        success_rate = 0.0

    if config.get("USE_WANDB", False):
        wandb.finish()

    def _save_network(result, dir_name):
        train_state = result["runner_state"].train_state
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_manager = CheckpointManager(dir_name, orbax_checkpointer, options)
        print(f"saved runner state to {dir_name}")
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            int(config["TOTAL_TIMESTEPS"]),
            train_state,
            save_kwargs={"save_args": save_args},
        )

    if config.get("MODULE_PATH"):
        directory, file_name = os.path.split(config["MODULE_PATH"])
        file_name_without_extension = os.path.splitext(file_name)[0]
        new_directory = os.path.join(
            directory, f"{file_name_without_extension}_policies"
        )
        os.makedirs(new_directory, exist_ok=True)
        _save_network(results[0], os.path.join(new_directory, "policies"))
        with open(f"{new_directory}/config.yaml", "w") as f:
            yaml.dump(config, f)
    else:
        raise AssertionError("should be in module path")

    out = results[0] if len(results) == 1 else {"per_seed": results}
    return out, success_rate > config["SUCCESS_STATE_RATE"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e7
    )
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
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)
    parser.add_argument("--module_path", type=str, default="/home/renos/flow-rl/exp/premade/1.py")
    parser.add_argument("--success_state_rate", type=float, default=0.8)
    parser.add_argument("--success_state", type=int)
    parser.add_argument(
        "--intrinsics-reward",
        action="store_true",
        help="Add intrinsics (hunger/thirst/fatigue) to reward",
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo_fast(args)
    else:
        with jax.disable_jit():
            run_ppo_fast(args)
