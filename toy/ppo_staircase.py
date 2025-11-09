"""PPO training for the Staircase Dungeon environment."""

from flowrl.wrappers import OptimisticResetVecEnvWrapper
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from staircase_env import StaircaseEnv, StaticEnvParams
from wrappers import LogWrapper
import wandb


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    static_params = StaticEnvParams()

    env_static_overrides = config.get("ENV_STATIC_OVERRIDES")
    if env_static_overrides:
        static_params = static_params.replace(**env_static_overrides)

    if config.get("USE_LONG_HORIZON_CORRIDOR"):
        corridor_length = max(3, int(config.get("CORRIDOR_LENGTH", 50)))
        static_params = static_params.replace(
            grid_height=1,
            grid_width=corridor_length,
            place_stairs_at_ends=True,
        )

    env = StaircaseEnv(static_params)
    env_params = env.default_params
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=config["NUM_ENVS"],
        reset_ratio=min(16, config["NUM_ENVS"]),
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, update_step = runner_state

                # SELECT ACTION
                rng, act_rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, step_rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    step_rng, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng, update_step)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, update_step = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

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

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Process metrics similar to ppo_flow
            def process_metric(x, returned_episode, is_special=False):
                if is_special:
                    # For per-floor metrics (one-hot encoded)
                    return (x * returned_episode[:, :, None]).sum(axis=(0, 1)) / returned_episode.sum()
                else:
                    return (x * returned_episode).sum() / returned_episode.sum()

            returned_episode = traj_batch.info["returned_episode"]
            metric = {
                key: process_metric(
                    value,
                    returned_episode,
                    is_special=(key == "floor_reached" or key == "floor_completed"),
                )
                for key, value in traj_batch.info.items()
            }
            metric["max_floor_reached"] = jnp.max(traj_batch.info["current_floor"])

            # Compute conditional probability: p(floor i+1 | floor i)
            floor_reached_prob = metric["floor_reached"]
            floor_conditional_prob = floor_reached_prob[1:] / (floor_reached_prob[:-1] + 1e-8)
            metric["floor_conditional_prob"] = floor_conditional_prob

            # Debugging mode
            if config.get("DEBUG"):
                def callback(info, ret_ep):
                    if ret_ep.sum() > 0:
                        return_values = info["returned_episode_returns"][ret_ep]
                        timesteps = info["timestep"][ret_ep] * config["NUM_ENVS"]
                        for t in range(len(timesteps)):
                            print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, traj_batch.info, returned_episode)

            # Wandb logging
            if config.get("USE_WANDB") and config.get("DEBUG"):
                def wandb_callback(info, update_step_val):
                    # Calculate global step
                    global_step = update_step_val * config["NUM_STEPS"] * config["NUM_ENVS"]

                    # Log to wandb
                    log_dict = {
                        "charts/episodic_return": info["returned_episode_returns"],
                        "charts/episodic_length": info["returned_episode_lengths"],
                        "charts/update_step": update_step_val,
                    }

                    # Add environment-specific metrics
                    if "current_floor" in info:
                        log_dict["env/avg_floor_reached"] = info["current_floor"]
                    if "max_floor_reached" in info:
                        log_dict["env/max_floor_reached"] = info["max_floor_reached"]
                    if "won" in info:
                        log_dict["env/win_rate"] = info["won"]
                    if "died" in info:
                        log_dict["env/death_rate"] = info["died"]

                    # Log per-floor completion rates
                    if "floor_completed" in info:
                        floor_completed = info["floor_completed"]
                        for floor in range(len(floor_completed)):
                            log_dict[f"floor_completion/floor_{floor}"] = floor_completed[floor]

                    # Log per-floor reached rates
                    if "floor_reached" in info:
                        floor_reached = info["floor_reached"]
                        for floor in range(len(floor_reached)):  # First 10 floors
                            log_dict[f"floor_reached/floor_{floor}"] = floor_reached[floor]

                    # Log conditional probability p(floor i+1 | floor i)
                    if "floor_conditional_prob" in info:
                        floor_cond_prob = info["floor_conditional_prob"]
                        for floor in range(len(floor_cond_prob)):  # First 10 transitions
                            log_dict[f"floor_conditional/p_floor_{floor+1}_given_{floor}"] = floor_cond_prob[floor]

                    wandb.log(log_dict, step=int(global_step))

                jax.debug.callback(wandb_callback, metric, update_step)

            runner_state = (train_state, env_state, last_obs, rng, update_step + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 0)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2e-4,
        "NUM_ENVS": 4096,
        "NUM_STEPS": 64,
        "TOTAL_TIMESTEPS": 1e9,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.05,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "USE_WANDB": True,
        "WANDB_PROJECT": "staircase-dungeon",
        "WANDB_ENTITY": None,  # Set to your wandb username/team
        # Environment configuration
        "USE_LONG_HORIZON_CORRIDOR": True,
        "CORRIDOR_LENGTH": 100,
        "ENV_STATIC_OVERRIDES": None,
    }

    # Initialize wandb
    if config["USE_WANDB"]:
        run_name = f"StaircaseDungeon-{int(config['TOTAL_TIMESTEPS'] // 1e6)}M"
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=run_name,
        ) 

    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    # Finish wandb
    if config["USE_WANDB"]:
        wandb.finish()

    print("Training completed!")
