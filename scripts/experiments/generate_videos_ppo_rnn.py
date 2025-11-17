#!/usr/bin/env python3
"""
Generate example videos from a PPO-RNN policy on Craftax.

Loads the PPO-RNN checkpoint, rolls out trajectories, renders frames, and saves MP4s.
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import cv2
import yaml

from flax.training.train_state import TrainState
import optax

from flowrl.ppo_rnn import ActorCriticRNN, ScannedRNN
from flowrl.wrappers import AutoResetEnvWrapper, BatchEnvWrapper
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_pixels as render_craftax_pixels_classic
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN as CLASSIC_BLOCK_PIXEL_SIZE
from craftax.craftax.renderer import render_craftax_pixels as render_craftax_pixels
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN as CRAFTAX_BLOCK_PIXEL_SIZE

from flowrl.utils.test import load_policy_params


def render_video(frames_rgb: np.ndarray, output_video: str, fps: float = 10.0):
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_video}")
    for frame in frames_rgb:
        # Convert to BGR for saving (creates contiguous array)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def generate_videos_ppo_rnn(policy_path: str,
                             output_dir: str = "videos_rnn",
                             env_name: str = "Craftax-Symbolic-v1",
                             num_videos: int = 3,
                             max_frames: int = 2000,
                             num_envs: int = 64,
                             fps: float = 10.0):
    os.makedirs(output_dir, exist_ok=True)

    # Load config (if present)
    cfg_path = os.path.join(policy_path, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_yaml = yaml.load(f, Loader=yaml.FullLoader)
        cfg = {k.upper(): v["value"] if isinstance(v, dict) and "value" in v else v for k, v in cfg_yaml.items()}
        layer_size = int(cfg.get("LAYER_SIZE", 512))
        env_name = cfg.get("ENV_NAME", env_name)
    else:
        layer_size = 512

    # Build environment (batched, auto-reset)
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)
    env_params = env.default_params

    # Construct RNN network + dummy optimizer (for TrainState wrapper)
    network = ActorCriticRNN(env.action_space(env_params).n, config={"LAYER_SIZE": layer_size})
    loaded = load_policy_params(policy_path)
    # loaded might be a TrainState-like pytree or dict; extract params
    params = loaded["params"] if isinstance(loaded, dict) and "params" in loaded else loaded.params
    tx = optax.adam(1e-4)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    @jax.jit
    def step_fn(rng, obs, env_state, hidden):
        # Add sequence dim=1 for RNN call
        obs_seq = obs[None, ...]
        dones_seq = jnp.zeros((1, obs.shape[0]), dtype=bool)
        new_hidden, pi, value = network.apply(train_state.params, hidden, (obs_seq, dones_seq))
        # Sample action then remove sequence dimension to match batched env: [num_envs]
        action = pi.sample(seed=rng)
        action = jnp.squeeze(action, axis=0)
        log_prob = pi.log_prob(action)
        # Env step
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        return rng, new_obs, new_env_state, new_hidden, action, reward, done

    # Choose renderer based on env
    use_classic = (env_name == "Craftax-Classic-Symbolic-v1")
    if use_classic:
        renderer = jax.jit(render_craftax_pixels_classic, static_argnums=(1,))
        block_px = CLASSIC_BLOCK_PIXEL_SIZE
    else:
        renderer = jax.jit(render_craftax_pixels, static_argnums=(1,))
        block_px = CRAFTAX_BLOCK_PIXEL_SIZE

    for vid in range(num_videos):
        print(f"\n=== Generating video {vid+1}/{num_videos} ===")
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        hidden = ScannedRNN.initialize_carry(num_envs, layer_size)

        frames_rgb = []
        rewards_arr = []

        # Rollout
        for t in range(max_frames):
            rng, obs, env_state, hidden, action, reward, done = step_fn(rng, obs, env_state, hidden)
            rewards_arr.append(np.array(reward))

            # Render a single randomly chosen env index for the video
            if t == 0:
                env_idx = np.random.randint(0, num_envs)
                print(f"Using environment {env_idx} for video frames")

            env_state_i = jax.tree.map(lambda x: x[env_idx], env_state)
            frame = renderer(env_state_i, block_px)
            # Force JAX to complete computation before numpy conversion
            frame = jax.device_get(frame)
            # Match play_craftax: upscale to ~64px tiles for readability
            scale = max(1, 64 // int(block_px))
            frame_rgb = np.array(frame, dtype=np.float32)
            if scale > 1:
                frame_rgb = np.repeat(np.repeat(frame_rgb, scale, axis=0), scale, axis=1)
            # Renderer outputs 0..255 floats; convert to uint8
            frame_u8 = np.clip(frame_rgb, 0, 255).astype(np.uint8)

            # Overlay reward text (per-step average reward across envs)
            # Keep in RGB format - will convert to BGR during write
            avg_r = float(jnp.mean(reward))
            cv2.putText(frame_u8, f"t={t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame_u8, f"avg_r={avg_r:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            frames_rgb.append(frame_u8)

        video_name = os.path.basename(os.path.normpath(policy_path)) + f"_rnn_video{vid+1}.mp4"
        out_path = os.path.join(output_dir, video_name)
        render_video(np.array(frames_rgb), out_path, fps=fps)
        print(f"✅ Saved video: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate videos from PPO-RNN policy")
    ap.add_argument("--policy_path", required=True, help="Path to policy (…/policies)")
    ap.add_argument("--output_dir", default="videos_rnn", help="Output directory for MP4s")
    ap.add_argument("--env_name", default="Craftax-Symbolic-v1", help="Environment name")
    ap.add_argument("--num_videos", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=2000)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    generate_videos_ppo_rnn(
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        env_name=args.env_name,
        num_videos=args.num_videos,
        max_frames=args.max_frames,
        num_envs=args.num_envs,
        fps=args.fps,
    )
