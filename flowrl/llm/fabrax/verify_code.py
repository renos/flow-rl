import os
# At the top of your file
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
import jax.numpy as jnp
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
from flowrl.wrappers import AutoResetEnvWrapper

from craftax.fabrax.envs.craftax_symbolic_env import (
    FabraxSymbolicEnv as FabraxEnv,
)
import numpy as np
from jax import jit



rng = jax.random.PRNGKey(np.random.randint(2**31))
rng, _rng = jax.random.split(rng)
env = AutoResetEnvWrapper(FabraxEnv())
env_params = env.default_params
obs, env_state = env.reset(_rng, env_params)
obsv, env_state, reward_e, done, info = env.step(_rng, env_state, 1)
state = env_state
player_intrinsics = jnp.array(
    [
        state.player_health,
        state.player_food,
        state.player_drink,
        state.player_energy,
    ]
)
intrinsics_diff = state.intrinsics_diff
inventory = state.inventory
inventory_diff = state.inventory_diff
closest_blocks = state.closest_blocks
closest_blocks_prev = state.closest_blocks_prev
health_penalty = 0
achievements = state.achievements
achievements_diff = state.achievements_diff


def verify_function(functions):
    # assuming functions have both reward and is_done
    namespace = {}
    exec("import jax", namespace)
    exec("import jax.numpy as jnp", namespace)
    exec("from craftax.fabrax.constants import *", namespace)
    exec("from craftax.fabrax.envs.craftax_state import Inventory", namespace)
    for function in functions:
        exec(function, namespace)

    is_task_done = jit(namespace["task_is_done"])
    task_reward = jit(namespace["task_reward"])
    is_task_done_sucessful = False
    is_task_done_exception = ""
    task_reward_sucessful = False
    task_reward_exception = ""
    batch_size = 2
    batched_inventory = jax.tree_map(
        lambda x: jnp.repeat(jnp.array([x]), batch_size, axis=0), inventory
    )
    batched_inventory_diff = jax.tree_map(
        lambda x: jnp.repeat(jnp.array([x]), batch_size, axis=0), inventory_diff
    )
    batched_closest_blocks = jnp.repeat(closest_blocks[None], batch_size, axis=0)
    batched_closest_blocks_prev = jnp.repeat(
        closest_blocks_prev[None], batch_size, axis=0
    )
    batched_player_intrinsics = jnp.repeat(player_intrinsics[None], batch_size, axis=0)
    batched_intrinsics_diff = jnp.repeat(intrinsics_diff[None], batch_size, axis=0)
    batched_achievements = jnp.repeat(achievements[None], batch_size, axis=0)
    batched_achievements_diff = jnp.repeat(achievements_diff[None], batch_size, axis=0)
    try:
        res = is_task_done(
            inventory,
            inventory_diff,
            closest_blocks,
            closest_blocks_prev,
            player_intrinsics,
            intrinsics_diff,
            achievements,
            1,  # n parameter
        )
        assert jnp.isscalar(res)
        # Test vmap with actual batched parameters
        vmapped_is_task_done = jax.vmap(is_task_done)

        batched_n = jnp.repeat(jnp.array([1]), batch_size, axis=0)
        
        vmapped_is_task_done(
            batched_inventory,
            batched_inventory_diff,
            batched_closest_blocks,
            batched_closest_blocks_prev,
            batched_player_intrinsics,
            batched_intrinsics_diff,
            batched_achievements,
            batched_n,
        )

        is_task_done_sucessful = True
    except Exception as e:
        is_task_done_exception = str(e)
    try:
        res = task_reward(
            inventory_diff,
            closest_blocks,
            closest_blocks_prev,
            intrinsics_diff,
            achievements_diff,
            health_penalty,
        )
        assert jnp.isscalar(res)
        vmapped_task_reward = jax.vmap(task_reward)
        batched_health_penalty = jnp.repeat(
            jnp.array([health_penalty]), batch_size, axis=0
        )

        vmapped_task_reward(
            batched_inventory_diff,
            batched_closest_blocks,
            batched_closest_blocks_prev,
            batched_intrinsics_diff,
            batched_achievements_diff,
            batched_health_penalty,
        )
        task_reward_sucessful = True
    except Exception as e:
        task_reward_exception = str(e)
        breakpoint()

    return (
        is_task_done_sucessful,
        is_task_done_exception,
        task_reward_sucessful,
        task_reward_exception,
    )
    # print(namespace["task_is_done"](inventory, inventory_diff, closest_blocks, closest_blocks_prev, player_intrinsics, intrinsics_diff, achievements, 1))
    # print(namespace["task_reward"](inventory_diff, closest_blocks, health_penalty))