import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
    }
    if "state_rates" in info:
        for i, state in enumerate(info["state_rates"]):
            to_log[f"state_rate_{i}"] = state
    if "reached_state" in info:
        for i, state in enumerate(info["reached_state"]):
            to_log[f"state_{i}_reached_prob"] = state

    # Per-skill diagnostics (log only if present)
    if "per_skill_kl" in info:
        for i, v in enumerate(info["per_skill_kl"]):
            to_log[f"per_skill_kl_{i}"] = v
            to_log[f"skill_{i}/kl"] = v
    if "per_skill_entropy" in info:
        for i, v in enumerate(info["per_skill_entropy"]):
            to_log[f"per_skill_entropy_{i}"] = v
            to_log[f"skill_{i}/entropy"] = v
    if "per_skill_value_loss" in info:
        for i, v in enumerate(info["per_skill_value_loss"]):
            to_log[f"per_skill_value_loss_{i}"] = v
            to_log[f"skill_{i}/value_loss"] = v
    if "per_skill_reward" in info:
        for i, v in enumerate(info["per_skill_reward"]):
            to_log[f"per_skill_reward_{i}"] = v
            to_log[f"skill_{i}/reward"] = v
    if "per_skill_adv_mean" in info:
        for i, v in enumerate(info["per_skill_adv_mean"]):
            to_log[f"per_skill_adv_mean_{i}"] = v
            to_log[f"skill_{i}/adv_mean"] = v
    if "per_skill_adv_std" in info:
        for i, v in enumerate(info["per_skill_adv_std"]):
            to_log[f"per_skill_adv_std_{i}"] = v
            to_log[f"skill_{i}/adv_std"] = v
    if "per_skill_counts" in info:
        for i, v in enumerate(info["per_skill_counts"]):
            to_log[f"per_skill_counts_{i}"] = v
            to_log[f"skill_{i}/count"] = v
    if "per_skill_episode_counts" in info:
        for i, v in enumerate(info["per_skill_episode_counts"]):
            to_log[f"per_skill_episode_counts_{i}"] = v
            to_log[f"skill_{i}/episode_count"] = v
    if "per_skill_success_rate" in info:
        for i, v in enumerate(info["per_skill_success_rate"]):
            to_log[f"per_skill_success_rate_{i}"] = v
            to_log[f"skill_{i}/success_rate"] = v
    if "per_skill_success_rate_batch" in info:
        for i, v in enumerate(info["per_skill_success_rate_batch"]):
            to_log[f"per_skill_success_rate_batch_{i}"] = v
            to_log[f"skill_{i}/success_rate_batch"] = v
    if "per_skill_successes" in info:
        for i, v in enumerate(info["per_skill_successes"]):
            to_log[f"per_skill_successes_{i}"] = v
            to_log[f"skill_{i}/successes"] = v
    if "per_skill_failures" in info:
        for i, v in enumerate(info["per_skill_failures"]):
            to_log[f"per_skill_failures_{i}"] = v
            to_log[f"skill_{i}/failures"] = v
    if "per_skill_total_transitions" in info:
        for i, v in enumerate(info["per_skill_total_transitions"]):
            to_log[f"per_skill_total_transitions_{i}"] = v
            to_log[f"skill_{i}/total_transitions"] = v

    # Add transition success metrics
    if "transition_success_rate" in info:
        to_log["transition_success_rate"] = info["transition_success_rate"]
    if "failure_transitions" in info:
        to_log["failure_transitions"] = info["failure_transitions"]
    if "success_transitions" in info:
        to_log["success_transitions"] = info["success_transitions"]

    # Progressive curriculum metrics
    if "progressive_reset_state" in info:
        to_log["progressive_reset_state"] = info["progressive_reset_state"]
    if "progressive_threshold_met" in info:
        to_log["progressive_threshold_met"] = info["progressive_threshold_met"]

    sum_achievements = 0
    for k, v in info.items():
        if "achievements" in k.lower():
            to_log[k] = v
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log


def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i][key]
                    if not jnp.isnan(val):
                        agg.append(val)

            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)

        log_times.append(time.time())

        if config["DEBUG"]:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)
