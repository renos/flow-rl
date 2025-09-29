#!/usr/bin/env python3
"""
Load clean trajectory data from pbz2 file and render as MP4 video (batched with JAX).
"""

import pickle
import bz2
import os
import sys
import cv2
import numpy as np
import jax
import jax.numpy as jnp
from Craftax.craftax.craftax.renderer import render_craftax_pixels
from tqdm import tqdm
from Craftax.craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
from pathlib import Path
from typing import Dict, Any, List
import time
# ---------------------------------------------------------------------
# Craftax types (import/define ONCE so pytrees share identical classes)
# ---------------------------------------------------------------------
try:
    # Some installs nest craftax twice; try that path first
    from Craftax.craftax.craftax.craftax_state import EnvState, Inventory, Mobs, Projectiles as _Projectiles
except Exception:
    try:
        from Craftax.craftax.craftax_state import EnvState, Inventory, Mobs, Projectiles as _Projectiles
    except Exception:
        # If Projectiles isn't exported in your version, define a stable fallback
        from Craftax.craftax.craftax.craftax_state import EnvState, Inventory, Mobs  # type: ignore
        from typing import NamedTuple
        class _Projectiles(NamedTuple):
            position: jnp.ndarray
            mask: jnp.ndarray
            type_id: jnp.ndarray

Projectiles = _Projectiles  # single, stable class for the whole run

# -------------------- I/O --------------------

def load_trajectory_data(file_path: str) -> Dict:
    """Load trajectory data from either .pkl or .pbz2 file"""
    if file_path.endswith('.pbz2'):
        with bz2.BZ2File(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return data

# -------------------- State reconstruction --------------------

def reconstruct_env_state(state_data: Dict[str, Any]) -> Any:
    # --- helpers
    def numpy_to_jax(x):
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        elif isinstance(x, dict):
            return {k: numpy_to_jax(v) for k, v in x.items()}
        return x
    as_i32 = lambda x: jnp.asarray(x, dtype=jnp.int32)
    as_b   = lambda x: jnp.asarray(x, dtype=jnp.bool_)

    jax_data = numpy_to_jax(state_data)

    # inventory & mobs (normalize dtypes a bit)
    inventory = Inventory(**jax_data['inventory'])

    melee_mobs = Mobs(
        position=as_i32(jax_data['melee_mobs']['position']),
        health=as_i32(jax_data['melee_mobs']['health']),
        mask=as_b(jax_data['melee_mobs']['mask']),
        attack_cooldown=as_i32(jax_data['melee_mobs']['attack_cooldown']),
        type_id=as_i32(jax_data['melee_mobs']['type_id'])
    )
    passive_mobs = Mobs(
        position=as_i32(jax_data['passive_mobs']['position']),
        health=as_i32(jax_data['passive_mobs']['health']),
        mask=as_b(jax_data['passive_mobs']['mask']),
        attack_cooldown=as_i32(jax_data['passive_mobs']['attack_cooldown']),
        type_id=as_i32(jax_data['passive_mobs']['type_id'])
    )
    ranged_mobs = Mobs(
        position=as_i32(jax_data['ranged_mobs']['position']),
        health=as_i32(jax_data['ranged_mobs']['health']),
        mask=as_b(jax_data['ranged_mobs']['mask']),
        attack_cooldown=as_i32(jax_data['ranged_mobs']['attack_cooldown']),
        type_id=as_i32(jax_data['ranged_mobs']['type_id'])
    )

    # ---------- PROPER PROJECTILES ----------
    num_levels = int(jax_data['map'].shape[0])
    slots = 1  # >=1 to avoid compile-time OOB when tracing scan body

    empty_proj = Projectiles(
        position=jnp.zeros((num_levels, slots, 2), dtype=jnp.int32),
        mask=jnp.zeros((num_levels, slots), dtype=jnp.bool_),
        type_id=jnp.zeros((num_levels, slots), dtype=jnp.int32),
    )
    empty_dirs = jnp.zeros((num_levels, slots, 2), dtype=jnp.int32)
    # ---------------------------------------

    # other small dummies the renderer doesn’t use directly
    dummy_plants_pos  = jnp.zeros((0, 3), dtype=jnp.int32)
    dummy_plants_age  = jnp.zeros((0,), dtype=jnp.int32)
    dummy_plants_mask = jnp.zeros((0,), dtype=jnp.bool_)
    dummy_potion_mapping = jnp.zeros((4,), dtype=jnp.int32)
    dummy_boss_timesteps = jnp.array(0, dtype=jnp.int32)
    dummy_rng = jax.random.PRNGKey(0)
    dummy_closest_blocks = jnp.zeros((9, 9), dtype=jnp.int32)
    dummy_state_diff = jnp.array(0, dtype=jnp.int32)

    env_state = EnvState(
        map=as_i32(jax_data['map']),
        item_map=as_i32(jax_data['item_map']),
        mob_map=as_b(jax_data['mob_map']),
        light_map=as_b(jax_data['light_map']),
        down_ladders=as_i32(jax_data['down_ladders']),
        up_ladders=as_i32(jax_data['up_ladders']),
        chests_opened=as_b(jax_data['chests_opened']),
        monsters_killed=as_i32(jax_data['monsters_killed']),
        player_position=as_i32(jax_data['player_position']),
        player_level=as_i32(jax_data['player_level']),
        player_direction=as_i32(jax_data['player_direction']),
        player_health=as_i32(jax_data['player_health']),
        player_food=as_i32(jax_data['player_food']),
        player_drink=as_i32(jax_data['player_drink']),
        player_energy=as_i32(jax_data['player_energy']),
        player_mana=as_i32(jax_data['player_mana']),
        is_sleeping=as_b(jax_data['is_sleeping']),
        is_resting=as_b(jax_data['is_resting']),
        player_recover=as_i32(jax_data['player_recover']),
        player_hunger=as_i32(jax_data['player_hunger']),
        player_thirst=as_i32(jax_data['player_thirst']),
        player_fatigue=as_i32(jax_data['player_fatigue']),
        player_recover_mana=as_i32(jax_data['player_recover_mana']),
        player_xp=as_i32(jax_data['player_xp']),
        player_dexterity=as_i32(jax_data['player_dexterity']),
        player_strength=as_i32(jax_data['player_strength']),
        player_intelligence=as_i32(jax_data['player_intelligence']),
        inventory=inventory,
        melee_mobs=melee_mobs,
        passive_mobs=passive_mobs,
        ranged_mobs=ranged_mobs,
        achievements=as_b(jax_data['achievements']),
        timestep=as_i32(jax_data['timestep']),
        light_level=jnp.asarray(jax_data['light_level'], dtype=jnp.float32),
        learned_spells=as_b(jax_data['learned_spells']),
        sword_enchantment=as_i32(jax_data['sword_enchantment']),
        bow_enchantment=as_i32(jax_data['bow_enchantment']),
        armour_enchantments=as_i32(jax_data['armour_enchantments']),
        boss_progress=as_i32(jax_data['boss_progress']),

        # ✅ projectiles & directions with correct shapes
        mob_projectiles=empty_proj,
        mob_projectile_directions=empty_dirs,
        player_projectiles=empty_proj,
        player_projectile_directions=empty_dirs,

        growing_plants_positions=dummy_plants_pos,
        growing_plants_age=dummy_plants_age,
        growing_plants_mask=dummy_plants_mask,
        potion_mapping=dummy_potion_mapping,
        boss_timesteps_to_spawn_this_round=dummy_boss_timesteps,
        state_rng=dummy_rng,
        closest_blocks=dummy_closest_blocks,
        player_state=dummy_state_diff,
        player_state_diff=dummy_state_diff,
        inventory_diff=dummy_state_diff,
        intrinsics_diff=dummy_state_diff,
        achievements_diff=dummy_state_diff,
        closest_blocks_prev=dummy_closest_blocks,
        task_done=jnp.array(False),
        fractal_noise_angles=(None, None, None, None),
    )
    return env_state

# -------------------- Batched rendering helpers --------------------

def stack_pytrees(pytrees: List[Any]) -> Any:
    """
    Stack a list of identical pytrees along a new leading axis (B, ...).
    Leaves that are None are returned as None without stacking.
    """
    return jax.tree_util.tree_map(
        lambda *xs: xs[0] if all(x is None for x in xs) else jnp.stack(xs, axis=0),
        *pytrees,
        is_leaf=lambda x: x is None
    )

def _render_one(s):
    return render_craftax_pixels(s, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)

_render_batch = jax.jit(jax.vmap(_render_one), donate_argnums=(0,))

# -------------------- Video rendering --------------------

def render_trajectory_video(
    trajectory_file: str,
    output_video: str,
    fps: float = 10.0,
    max_frames: int = None,
    skip_frames: int = 1,
    batch_size: int = 64,
) -> None:
    print(f"Loading trajectory from: {trajectory_file}")
    trajectory_data = load_trajectory_data(trajectory_file)

    states = trajectory_data['states']
    actions = trajectory_data['actions']
    rewards = trajectory_data['rewards']
    metadata = trajectory_data['metadata']

    n_states_total = len(states)

    actions_np = np.asarray(actions)
    if actions_np.shape[0] < n_states_total:
        pad = n_states_total - actions_np.shape[0]
        # repeat last action; if there are zero actions, pad zeros
        if actions_np.size == 0:
            actions_np = np.zeros((n_states_total,), dtype=np.int32)
        else:
            actions_np = np.pad(actions_np, (0, pad), mode="edge")

    rewards_np = np.asarray(rewards, dtype=np.float32)
    if rewards_np.shape[0] < n_states_total:
        pad = n_states_total - rewards_np.shape[0]
        if rewards_np.size == 0:
            rewards_np = np.zeros((n_states_total,), dtype=np.float32)
        else:
            # choose one: repeat last reward (edge) or pad zeros; zeros is often nicer
            rewards_np = np.pad(rewards_np, (0, pad), mode="constant", constant_values=0.0)

    print("Trajectory info:")
    print(f"  Total timesteps: {metadata['num_timesteps']}")
    print(f"  Map shape: {metadata['map_shape']}")
    print(f"  Total reward: {metadata['total_reward']:.4f}")

    num_states = len(states)
    if max_frames is not None:
        num_states = min(num_states, max_frames)

    frame_indices = list(range(0, num_states, skip_frames))
    actual_frames = len(frame_indices)
    print(f"Rendering {actual_frames} frames (skip={skip_frames}) at {fps} fps, batch_size={batch_size}")

    print("Rendering first frame to get dimensions / JIT warmup...")
    first_state = reconstruct_env_state(states[frame_indices[0]])
    first_frame = np.array(_render_one(first_state)).astype(np.uint8)
    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width} x {height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_video}")

    print("Rendering frames in batches...")
    try:
        _ = _render_batch(stack_pytrees([first_state]))  # tiny warmup

        for batch_start in tqdm(range(0, actual_frames, batch_size)):
            batch_idxs = frame_indices[batch_start: batch_start + batch_size]
            t0 = time.time()
            batch_states = [reconstruct_env_state(states[idx]) for idx in batch_idxs]
            stacked = stack_pytrees(batch_states)   # (B, ...)
            t1= time.time()
            print(time.time()-t0)

            batch_frames = _render_batch(stacked)
            print(f"Rendering time {time.time()-t1}")
            batch_frames = np.array(batch_frames).astype(np.uint8)

            for idx, frame in zip(batch_idxs, batch_frames):
                act = int(actions_np[idx])
                rew = float(rewards_np[idx])
                frame_with_info = add_info_overlay(frame, idx, act, rew, states[idx])
                out.write(cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR))

            del batch_states, stacked, batch_frames
    finally:
        out.release()

    print(f"✅ Video saved to: {output_video}")
    print(f"  Duration: {actual_frames / fps:.1f} seconds")
    print(f"  File size: {os.path.getsize(output_video) / (1024*1024):.1f} MB")

# -------------------- Overlay --------------------

def add_info_overlay(frame: np.ndarray, timestep: int, action: int, reward: float, state_data: Dict) -> np.ndarray:
    frame_with_info = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    line_height = 20

    info_lines = [
        f"Step: {timestep}",
        f"Action: {action}",
        f"Reward: {reward:.3f}",
        f"Health: {int(state_data['player_health'])}",
        f"Level: {int(state_data['player_level'])}",
        f"Pos: ({int(state_data['player_position'][0])}, {int(state_data['player_position'][1])})",
    ]
    wood = state_data['inventory']['wood']
    stone = state_data['inventory']['stone']
    if wood > 0 or stone > 0:
        info_lines.append(f"Wood: {int(wood)}, Stone: {int(stone)}")

    y = 25
    for line in info_lines:
        cv2.putText(frame_with_info, line, (10, y), font, font_scale, color, thickness)
        y += line_height
    return frame_with_info

# -------------------- CLI --------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render trajectory video from clean pbz2 file (batched)")
    parser.add_argument("trajectory_file", help="Path to trajectory file (.pkl or .pbz2)")
    parser.add_argument("-o", "--output", help="Output MP4 file path", default=None)
    parser.add_argument("--fps", type=float, default=10.0, help="Video FPS (default: 10)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to render")
    parser.add_argument("--skip", type=int, default=1, help="Skip every N frames (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for JAX rendering (default: 64)")
    args = parser.parse_args()

    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        sys.exit(1)

    if args.output is None:
        trajectory_path = Path(args.trajectory_file)
        args.output = str(trajectory_path.with_suffix('.mp4'))

    render_trajectory_video(
        args.trajectory_file,
        args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        skip_frames=args.skip,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
