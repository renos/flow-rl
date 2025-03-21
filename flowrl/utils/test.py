import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from flowrl.models.actor_critic import ActorCriticMoE
import jax
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManager,
    CheckpointManagerOptions,
)
from glob import glob
import yaml
import cv2
import numpy as np

# from environment_base.wrappers import (
#     LogWrapper,
#     OptimisticResetVecEnvWrapper,
#     AutoResetEnvWrapper,
#     BatchEnvWrapper,
# )
from flowrl.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from typing import NamedTuple
import jax.numpy as jnp
from logz.batch_logging import batch_log, create_log_dict
from collections import defaultdict

import importlib


class Transition(NamedTuple):
    done: jnp.ndarray
    player_state: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def load_policy_params(path, verbose=False):
    # Step 1: Load the Policy Parameters
    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    policies_dir = f"{path}/policies"

    try:
        # Find the checkpoint with the largest timestep using max() and a generator expression
        latest_checkpoint = max(
            glob(policies_dir + "/*"), key=lambda cp: float(cp.split("/")[-1])
        )
        largest_timestep = latest_checkpoint.split("/")[-1]
        if verbose:
            print(
                f"The latest checkpoint is at: {latest_checkpoint} with timestep {largest_timestep}"
            )
    except ValueError:
        # This handles the case where checkpoint names don't convert to float properly
        print("Checkpoint names are not properly formatted as floats.")
    except Exception as e:
        # Handle other exceptions, such as no files matching the glob pattern
        print(f"No valid checkpoints found. Error: {e}")

    checkpoint_manager = CheckpointManager(policies_dir, orbax_checkpointer, options)
    train_state = checkpoint_manager.restore(largest_timestep)
    return train_state


def restore_model(path, env, num_tasks, hierarchical=False, bottomup=False):
    config_path = f"{path}/config.yaml"
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

    train_state = load_policy_params(path)

    return network, train_state


def render_video(frames, states, video_path):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

    for frame, state in zip(np.array(frames).astype(np.uint8), states):
        # Your frames must be BGR to save correctly
        # If your frames are not in BGR format (e.g., RGB), convert them
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # out.write(frame_bgr)
        frame_with_text = frame.copy()
        cv2.putText(
            frame_with_text,
            f"State: {int(state)}",
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Font scale
            (255, 255, 255),  # Color (white)
            2,  # Thickness
        )

        # Convert to BGR for saving
        frame_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release everything when the job is finished
    out.release()


import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import (
    OBS_DIM,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
    BLOCK_PIXEL_SIZE_HUMAN,
)
from flowrl.wrappers import AutoResetEnvWrapper

from craftax.craftax_classic.renderer import render_craftax_pixels
import numpy as np


def gen_frames_hierarchical(policy_path, max_num_frames=2000):
    config_path = f"{policy_path}/config.yaml"
    with open(config_path, "r") as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
    config = {
        k.upper(): v["value"] if type(v) == dict and "value" in v else v
        for k, v in config_.items()
    }

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    print(rng)
    rng, _rng = jax.random.split(rng)
    # env = AutoResetEnvWrapper(
    #     CraftaxClassicSymbolicEnv(CraftaxClassicSymbolicEnv.default_static_params())
    # )
    # initialize env
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

    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_env import make_craftax_flow_env_from_name

        env = make_craftax_flow_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"], module_dict
        )
        env = AutoResetEnvWrapper(env)

    env_params = env.default_params
    obs, env_state = env.reset(_rng, env_params)
    task_to_skill_index = jnp.array(env.task_to_skill_index)
    num_tasks_ = len(task_to_skill_index)

    # load model
    network, train_state = restore_model(policy_path, env, num_tasks_)

    def map_player_state_to_skill(player_state):
        # Create a one-hot encoding of task_to_skill_index
        player_state_one_hot = jnp.eye(num_tasks_)[player_state]
        player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)
        return player_skill

    renderer = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def step_fn(carry, x):
        rng, obs, env_state, i, continue_loop = carry
        rng, _rng = jax.random.split(rng)

        new_obs, new_env_state, frame, output = obs, env_state, None, None

        player_state = jnp.expand_dims(env_state.player_state, 0)
        player_skill = map_player_state_to_skill(player_state)

        pi, value = network.apply(
            train_state["params"],
            jnp.expand_dims(obs, 0),
            player_skill,
        )
        action = pi.sample(seed=_rng)[0]

        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env.step(
            _rng, env_state, action, env_params
        )

        frame = renderer(new_env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        output = (frame, new_env_state.player_state, new_env_state, action)

        new_carry = (rng, new_obs, new_env_state, i + 1, continue_loop)

        return new_carry, output

    # Initial carry state, including a flag to indicate whether to continue the loop
    initial_carry = (rng, obs, env_state, 0, True)

    # Dummy input array for scan, its length determines the number of iterations
    inputs = jnp.arange(max_num_frames)

    # Execute the scan
    _, output = jax.lax.scan(step_fn, initial_carry, inputs)

    frames, states, new_env_states, actions = output
    return frames, states, new_env_states, actions


def evaluate_hierarchical(
    network, train_state, max_num_frames=2000, num_envs=128, num_episodes=1000
):

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    env_static_params = CraftaxClassicSymbolicEnv.default_static_params()
    env = CraftaxClassicSymbolicEnv(env_static_params)
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    env_params = env.default_params.replace(max_timesteps=max_num_frames)
    obs, env_state = env.reset(_rng, env_params)
    runner_state = (
        train_state,
        env_state,
        obs,
        env_state,
        _rng,
        0,
    )

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
        rng, _rng = jax.random.split(rng)
        player_state = env_state.env_state.player_state
        pi, value = network.apply(train_state["params"], last_obs, player_state)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward_e, done, info = env.step(
            _rng, env_state, action, env_params
        )
        done = info["task_done"]

        reward = reward_e
        reward_i = reward_e

        transition = Transition(
            done=done,
            player_state=player_state,
            action=action,
            value=value,
            reward=reward,
            reward_i=reward_i,
            reward_e=reward_e,
            log_prob=log_prob,
            obs=last_obs,
            next_obs=obsv,
            info=info,
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

    ran_episodes = 0
    results_dict = defaultdict(list)

    while ran_episodes < num_episodes:

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, max_num_frames
        )

        def process(x, returned_episode, is_special=False):
            if is_special:
                # Reshape returned_episode to match the dimensions of x for broadcasting
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
            )
            for key, value in traj_batch.info.items()
        }
        player_state_one_hot = jax.nn.one_hot(traj_batch.player_state, num_classes=12)
        state_occurrences = player_state_one_hot.sum(
            axis=(0, 1)
        )  # Sum over batch and timesteps
        total_weights = traj_batch.info["returned_episode"].sum()
        state_rates = state_occurrences / total_weights
        metric["state_rates"] = state_rates
        config = {}
        config["ENV_NAME"] = "Craftax-Classic-v1"
        to_log = create_log_dict(metric, config)
        to_log["num_episodes"] = total_weights
        for key, value in to_log.items():
            if key == "num_episodes":
                results_dict[key].append(value)
            else:
                results_dict[key].append(value * total_weights)
        ran_episodes += total_weights
    normed_result_dict = {}
    num_episodes = np.sum(results_dict["num_episodes"])
    normed_result_dict["num_episodes"] = num_episodes
    for key, value in results_dict.items():
        normed_result_dict[key] = np.sum(value) / num_episodes
    return normed_result_dict
