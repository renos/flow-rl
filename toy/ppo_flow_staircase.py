"""Flow-style PPO trainer for the Staircase Dungeon environment.

This variant follows the flow-based multi-skill setup used in `ppo_flow_w_reset`:
- Maintains a separate policy/value head per skill (here: per floor).
- Tracks the best state reached for each environment and performs progressive resets.
- Advances the training frontier once per-floor success rates surpass a threshold.
"""

import argparse
from typing import Any, Dict, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import wandb

from staircase_env import StaircaseEnv, StaticEnvParams
from wrappers import LogWrapper


class ActorCriticPerSkill(nn.Module):
    """Actor-critic with dedicated parameters per skill (floor)."""

    action_dim: int
    num_skills: int
    hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "tanh"

    def setup(self):
        self.act_fn = nn.relu if self.activation == "relu" else nn.tanh
        self.actor_networks = tuple(
            self.create_network(is_actor=True, name=f"actor_network_{i}")
            for i in range(self.num_skills)
        )
        self.critic_networks = tuple(
            self.create_network(is_actor=False, name=f"critic_network_{i}")
            for i in range(self.num_skills)
        )

    def create_network(self, is_actor: bool, name: str) -> nn.Module:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        layers = []
        for _ in range(max(self.num_layers - 1, 0)):
            layers.append(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                )
            )
            layers.append(activation)
        layers.append(
            nn.Dense(
                self.action_dim if is_actor else 1,
                kernel_init=orthogonal(0.01 if is_actor else 1.0),
                bias_init=constant(0.0),
            )
        )
        return nn.Sequential(layers, name=name)

    def __call__(self, obs: jnp.ndarray, skill_ids: jnp.ndarray):
        # Compute logits/value for every skill head, then select with one-hot gating
        actor_outputs = jnp.stack(
            [network(obs) for network in self.actor_networks], axis=1
        )  # [B, num_skills, action_dim]
        critic_outputs = jnp.stack(
            [network(obs) for network in self.critic_networks], axis=1
        )  # [B, num_skills, 1]

        gate = jax.nn.one_hot(skill_ids, self.num_skills, dtype=obs.dtype)
        selected_actor = jnp.sum(actor_outputs * gate[:, :, None], axis=1)
        selected_value = jnp.sum(critic_outputs * gate[:, :, None], axis=1).squeeze(-1)

        pi = distrax.Categorical(logits=selected_actor)
        return pi, selected_value


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Dict[str, Any]
    floor: jnp.ndarray


def apply_mask(mask: jnp.ndarray, new: jnp.ndarray, old: jnp.ndarray) -> jnp.ndarray:
    while mask.ndim < new.ndim:
        mask = mask[..., None]
    return jnp.where(mask, new, old)


def mask_tree(mask: jnp.ndarray, new_tree: Any, old_tree: Any) -> Any:
    return jax.tree.map(lambda new, old: apply_mask(mask, new, old), new_tree, old_tree)


def make_train(config):
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    base_pattern = jnp.array(
        [
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
        ],
        dtype=bool,
    )

    num_floors_cfg = int(config.get("NUM_FLOORS", 30))
    if base_pattern.shape[0] >= num_floors_cfg:
        stair_pattern = base_pattern[:num_floors_cfg]
    else:
        repeats = int(np.ceil(num_floors_cfg / base_pattern.shape[0]))
        stair_pattern = jnp.tile(base_pattern, (repeats,))[:num_floors_cfg]

    static_params = StaticEnvParams(
        num_floors=num_floors_cfg,
        grid_height=int(config.get("GRID_HEIGHT", 10)),
        grid_width=int(config.get("GRID_WIDTH", 10)),
        place_stairs_at_ends=bool(config.get("PLACE_STAIRS_AT_ENDS", False)),
        correct_stair_pattern=stair_pattern,
    )

    env = LogWrapper(StaircaseEnv(static_params=static_params))
    env_params = env.default_params
    num_floors = env._env.static_params.num_floors

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    if config.get("USE_WANDB", False):
        num_env_steps = config["NUM_ENVS"] * config["NUM_STEPS"]

        def wandb_callback(metrics, update_idx):
            update_idx = int(update_idx)
            global_step = (update_idx + 1) * num_env_steps
            log_dict: Dict[str, Any] = {}
            for key, value in metrics.items():
                np_value = np.asarray(value)
                if np_value.ndim == 0:
                    log_dict[key] = np_value.item()
                else:
                    named = key in {"floor_reached", "floor_completed", "success/ema", "success/batch", "success/count"}
                    for idx, item in enumerate(np_value.tolist()):
                        if named:
                            log_dict[f"{key}/floor_{idx}"] = float(item)
                        else:
                            log_dict[f"{key}/{idx}"] = float(item)
            wandb.log(log_dict, step=global_step)
    else:
        wandb_callback = None

    def train(rng):
        network = ActorCriticPerSkill(
            action_dim=env.action_space(env_params).n,
            num_skills=num_floors,
            hidden_dim=config["LAYER_SIZE"],
            num_layers=config["NUM_LAYERS"],
            activation=config["ACTIVATION"],
        )
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((config["NUM_ENVS"], env.observation_space(env_params).shape[0]))
        dummy_skill = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        params = network.init(init_rng, dummy_obs, dummy_skill)

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
            params=params,
            tx=tx,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)

        ex_state = {
            "best_floor": env_state.env_state.current_floor,
            "snapshot_obs": obs,
            "snapshot_state": env_state.env_state,
            "success_ema": jnp.zeros((num_floors,), dtype=jnp.float32),
            "progressive_floor": jnp.array(0, dtype=jnp.int32),
        }

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, ex_state, rng, update_step = runner_state

            floor = env_state.env_state.current_floor
            rng, act_rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, last_obs, floor)
            action = pi.sample(seed=act_rng)
            log_prob = pi.log_prob(action)

            rng, step_rng = jax.random.split(rng)
            step_keys = jax.random.split(step_rng, config["NUM_ENVS"])
            next_obs, next_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, env_state, action, env_params)

            next_floor = next_state.env_state.current_floor

            improved = next_floor > ex_state["best_floor"]
            snapshot_obs = apply_mask(improved, next_obs, ex_state["snapshot_obs"])
            snapshot_state = mask_tree(improved, next_state.env_state, ex_state["snapshot_state"])
            best_floor = apply_mask(improved, next_floor, ex_state["best_floor"])

            progressive_floor = ex_state["progressive_floor"]
            frontier = jnp.broadcast_to(progressive_floor, floor.shape)
            has_snapshot = best_floor >= floor
            non_terminal = ~info.get("won", jnp.zeros_like(done, dtype=bool))
            restore_mask = done & non_terminal & (floor >= frontier) & has_snapshot

            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, config["NUM_ENVS"])
            reset_obs, reset_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
            fresh_mask = done & ~restore_mask

            merged_obs = apply_mask(restore_mask, snapshot_obs, next_obs)
            merged_obs = apply_mask(fresh_mask, reset_obs, merged_obs)

            merged_env_state = next_state.env_state
            merged_env_state = mask_tree(restore_mask, snapshot_state, merged_env_state)
            merged_env_state = mask_tree(fresh_mask, reset_state.env_state, merged_env_state)

            episode_returns = apply_mask(done, jnp.zeros_like(next_state.episode_returns), next_state.episode_returns)
            episode_lengths = apply_mask(done, jnp.zeros_like(next_state.episode_lengths), next_state.episode_lengths)
            timestep = apply_mask(done, jnp.zeros_like(next_state.timestep), next_state.timestep)

            merged_state = next_state.replace(
                env_state=merged_env_state,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
                timestep=timestep,
            )

            snapshot_obs = apply_mask(fresh_mask, reset_obs, snapshot_obs)
            snapshot_state = mask_tree(fresh_mask, reset_state.env_state, snapshot_state)
            best_floor = apply_mask(fresh_mask, reset_state.env_state.current_floor, best_floor)

            ex_state = {
                **ex_state,
                "best_floor": best_floor,
                "snapshot_obs": snapshot_obs,
                "snapshot_state": snapshot_state,
            }

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                next_obs=merged_obs,
                info=info,
                floor=floor,
            )

            runner_state = (
                train_state,
                merged_state,
                merged_obs,
                ex_state,
                rng,
                update_step,
            )
            return runner_state, transition

        def _update_step(runner_state, _):
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            train_state, env_state, last_obs, ex_state, rng, update_step = runner_state

            state_skill = env_state.env_state.current_floor
            _, last_val = network.apply(train_state.params, last_obs, state_skill)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = transition.reward + config["GAMMA"] * next_value * (1 - transition.done) - transition.value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(carry, _):
                train_state, traj_batch, advantages, targets, rng = carry
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    shuffled,
                )

                def _update_minbatch(train_state, mini):
                    traj_batch, gae, targets = mini

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs, traj_batch.floor)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    (_, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params, traj_batch, gae, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, aux

                train_state, aux_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), aux_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Aggregate per-floor success/failure statistics for progressive frontier updates
            floor_indices = traj_batch.floor.astype(jnp.int32)
            floor_one_hot = jax.nn.one_hot(floor_indices, num_floors)
            successes = traj_batch.info["floor_completed"].astype(jnp.float32)
            failures = floor_one_hot * traj_batch.info["died"].astype(jnp.float32)[..., None]

            per_skill_success = successes.sum(axis=(0, 1))
            per_skill_failure = failures.sum(axis=(0, 1))
            per_skill_total = per_skill_success + per_skill_failure

            prev_ema = ex_state["success_ema"]
            batch_rate = jnp.where(
                per_skill_total > 0,
                per_skill_success / (per_skill_total + 1e-8),
                prev_ema,
            )
            ema = jnp.where(
                per_skill_total > 0,
                (1.0 - config["SUCCESS_RATE_EMA_ALPHA"]) * prev_ema
                + config["SUCCESS_RATE_EMA_ALPHA"] * batch_rate,
                prev_ema,
            )

            current_frontier = ex_state["progressive_floor"]
            current_rate = ema[current_frontier]
            can_advance = jnp.logical_and(
                current_rate >= config["PROGRESSIVE_THRESHOLD"],
                current_frontier < num_floors - 1,
            )
            new_frontier = jnp.where(
                can_advance,
                current_frontier + 1,
                current_frontier,
            )
            max_best = jnp.maximum(jnp.max(ex_state["best_floor"]), current_frontier)
            new_frontier = jnp.minimum(new_frontier, max_best)

            ex_state = {
                **ex_state,
                "success_ema": ema,
                "progressive_floor": new_frontier,
            }

            value_losses, policy_losses, entropy_terms = loss_info
            metric = {
                "loss/value": value_losses.mean(),
                "loss/policy": policy_losses.mean(),
                "loss/entropy": entropy_terms.mean(),
                "progressive/frontier": new_frontier,
                "progressive/success_rate": current_rate,
            }

            returned_episode = traj_batch.info["returned_episode"]

            def process_metric(x, returned_episode, is_vector=False):
                if is_vector:
                    return (x * returned_episode[..., None]).sum(axis=(0, 1)) / jnp.maximum(
                        returned_episode.sum(), 1.0
                    )
                return (x * returned_episode).sum() / jnp.maximum(returned_episode.sum(), 1.0)

            metric.update(
                {
                    key: process_metric(
                        value,
                        returned_episode,
                        is_vector=key in {"floor_reached", "floor_completed"},
                    )
                    for key, value in traj_batch.info.items()
                }
            )
            metric["success/ema"] = ema
            metric["success/batch"] = batch_rate
            metric["success/count"] = per_skill_total

            if wandb_callback is not None:
                jax.debug.callback(wandb_callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, loop_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, ex_state, loop_rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main():
    parser = argparse.ArgumentParser(description="Flow-style PPO for Staircase Dungeon")
    parser.add_argument("--total-timesteps", type=float, default=1e8)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--layer-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--progressive-threshold", type=float, default=0.3)
    parser.add_argument("--success-rate-ema-alpha", type=float, default=0.2)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="staircase-dungeon")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "NUM_MINIBATCHES": args.num_minibatches,
        "UPDATE_EPOCHS": args.update_epochs,
        "LR": args.lr,
        "GAE_LAMBDA": args.gae_lambda,
        "GAMMA": args.gamma,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "LAYER_SIZE": args.layer_size,
        "NUM_LAYERS": args.num_layers,
        "ACTIVATION": args.activation,
        "ANNEAL_LR": args.anneal_lr,
        "PROGRESSIVE_THRESHOLD": args.progressive_threshold,
        "SUCCESS_RATE_EMA_ALPHA": args.success_rate_ema_alpha,
        "USE_WANDB": args.use_wandb,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
    }

    wandb_run = None
    if config["USE_WANDB"]:
        run_name = args.wandb_run_name or f"ppo-flow-staircase-{int(args.total_timesteps // 1e6)}M"
        wandb_run = wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=run_name,
        )

    train_fn = jax.jit(make_train(config))
    rng = jax.random.PRNGKey(args.seed)
    train_fn(rng)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
