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
from .models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
    ActorCriticMoE,
)
from .wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
import importlib

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    player_state: jnp.ndarray
    inventory: jnp.ndarray


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
        network = ActorCriticMoE(
            action_dim=env.action_space(env_params).n,
            layer_width=config["LAYER_SIZE"],
            num_layers=4,
            num_tasks=num_heads,
        )
        return network

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            # network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
            network = ActorCriticMoE(
                action_dim=env.action_space(env_params).n,
                layer_width=config["LAYER_SIZE"],
                num_layers=4,
                num_tasks=num_heads,
            )
        else:
            # network = ActorCriticConv(
            #     env.action_space(env_params).n, config["LAYER_SIZE"]
            # )
            assert 0, "Not implemented atm"

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        init_states = jnp.zeros((1,), dtype=jnp.int32)
        network_params = network.init(_rng, init_x, init_states)
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

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                player_state = env_state.env_state.player_state
                state_skill = map_player_state_to_skill(player_state)
                inv = env_state.env_state.inventory

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs, state_skill)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                done = info["task_done"]

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                    player_state=player_state,
                    inventory=inv,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            state_skill = map_player_state_to_skill(env_state.env_state.player_state)
            _, last_val = network.apply(train_state.params, last_obs, state_skill)

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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        state_skill = map_player_state_to_skill(traj_batch.player_state)
                        pi, value = network.apply(params, traj_batch.obs, state_skill)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
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
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]

            def process(x, returned_episode, is_special=False, is_nearest=False):
                # reached_ep = returned_episode
                reached_ep = traj_batch.player_state == config["SUCCESS_STATE_INDEX"]
                if is_nearest:
                    # mask_seen = (x[:, :, :, 0, :] < 30) and (x[:, :, :, 1, :] < 30)
                    mask_seen = jnp.logical_and(
                        x[:, :, :, 0, :] < 30, x[:, :, :, 1, :] < 30
                    )
                    return (mask_seen * reached_ep[:, :, None, None]).sum(
                        axis=(0, 1)
                    ) / reached_ep.sum()
                elif is_special:
                    # Reshape returned_episode to match the dimensions of x for broadcasting
                    # also we want returned episode to track the success rates directly
                    return (x * returned_episode[:, :, None]).sum(
                        axis=(0, 1)
                    ) / returned_episode.sum()
                else:
                    return (x * reached_ep).sum() / reached_ep.sum()

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

            # inventory when done
            # inventory = jnp.array(
            #     [
            #         traj_batch.inventory.wood,
            #         traj_batch.inventory.stone,
            #         traj_batch.inventory.coal,
            #         traj_batch.inventory.iron,
            #         traj_batch.inventory.diamond,
            #         traj_batch.inventory.sapling,
            #         traj_batch.inventory.wood_pickaxe,
            #         traj_batch.inventory.stone_pickaxe,
            #         traj_batch.inventory.iron_pickaxe,
            #         traj_batch.inventory.wood_sword,
            #         traj_batch.inventory.stone_sword,
            #         traj_batch.inventory.iron_sword,
            #     ]
            # ).astype(jnp.float16)
            flat_values, _ = jax.tree_util.tree_flatten(traj_batch.inventory)
            # Convert the flattened values to a single array by stacking them
            inventory = jnp.stack(flat_values, axis=0).astype(jnp.float16)
            # done_point = traj_batch.done
            done_point = traj_batch.player_state == config["SUCCESS_STATE_INDEX"]
            avg_item = (
                jax.nn.one_hot(inventory, num_classes=10) * done_point[None, ..., None]
            ).sum(axis=(-3, -2)) / done_point.sum()
            metric["average_item"] = avg_item

            rng = update_state[-1]

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
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )

        # runner_state, metric = jax.lax.scan(
        #     _update_step, runner_state, None, config["NUM_UPDATES"]
        # )
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
        }  # , "info": metric}

    return train


def run_ppo(config, training_state_i=2):
    config = {k.upper(): v for k, v in config.__dict__.items()}

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
        with open(f"{new_directory}/config.yaml", "w") as f:
            yaml.dump(config, f)
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
    parser.add_argument("--gae_lambda", type=float, default=0.8)
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

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    # Flow Params
    parser.add_argument(
        "--module_path", type=str, default="/home/renos/flow-rl/exp/premade/1.py"
    )
    # what sucdess rate to achieve before optimizing next node
    parser.add_argument("--success_state_rate", type=float, default=0.8)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    # import jax
    # from jax import profiler

    # 1. Create a trace directory to save profiling results

    # trace_dir = "/home/renos/tensorboard2"
    # with profiler.trace(trace_dir, create_perfetto_link=True):
    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
