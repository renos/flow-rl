#!/usr/bin/env python3
"""
Simple test script to understand Craftax renderer output format.
"""
import jax
import jax.numpy as jnp
import numpy as np
import cv2

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN

# Create environment and get initial state
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

# Reset to get initial state
rng = jax.random.PRNGKey(0)
obs, env_state = env.reset(rng, env_params)

# Render a frame
block_px = BLOCK_PIXEL_SIZE_HUMAN
frame = render_craftax_pixels(env_state, block_pixel_size=block_px)

# Print debugging info
print(f"Frame type: {type(frame)}")
print(f"Frame dtype: {frame.dtype}")
print(f"Frame shape: {frame.shape}")
print(f"Frame min value: {jnp.min(frame)}")
print(f"Frame max value: {jnp.max(frame)}")
print(f"Frame is JAX array: {isinstance(frame, jnp.ndarray)}")

# Convert to numpy for testing
frame_np = np.array(frame)
print(f"\nNumpy array dtype: {frame_np.dtype}")
print(f"Numpy array min: {frame_np.min()}")
print(f"Numpy array max: {frame_np.max()}")

# Test different conversion approaches
print("\n=== Testing conversion approaches ===")

# Approach 1: Direct uint8 conversion (what the working RNN version does)
print("\n1. Direct uint8 conversion:")
frame_uint8_direct = np.array(frame, dtype=np.uint8)
print(f"   dtype: {frame_uint8_direct.dtype}, min: {frame_uint8_direct.min()}, max: {frame_uint8_direct.max()}")

# Approach 2: Float32 + clip + uint8 (what your current version does)
print("\n2. Float32 -> clip -> uint8:")
frame_float32 = np.array(frame, dtype=np.float32)
frame_uint8_clipped = np.clip(frame_float32, 0, 255).astype(np.uint8)
print(f"   dtype: {frame_uint8_clipped.dtype}, min: {frame_uint8_clipped.min()}, max: {frame_uint8_clipped.max()}")

# Test if they're the same
print(f"\n3. Are they identical? {np.array_equal(frame_uint8_direct, frame_uint8_clipped)}")
print(f"   Max difference: {np.abs(frame_uint8_direct.astype(int) - frame_uint8_clipped.astype(int)).max()}")

# Test saving with both methods
print("\n=== Testing video write compatibility ===")

# Method 1: Direct
cv2.imwrite("test_direct.png", cv2.cvtColor(frame_uint8_direct, cv2.COLOR_RGB2BGR))
print("Saved test_direct.png")

# Method 2: With clip
cv2.imwrite("test_clipped.png", cv2.cvtColor(frame_uint8_clipped, cv2.COLOR_RGB2BGR))
print("Saved test_clipped.png")

# Test video writer with both
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out1 = cv2.VideoWriter("test_direct.mp4", fourcc, 10.0, (frame_uint8_direct.shape[1], frame_uint8_direct.shape[0]))
out1.write(cv2.cvtColor(frame_uint8_direct, cv2.COLOR_RGB2BGR))
out1.release()
print("Saved test_direct.mp4")

out2 = cv2.VideoWriter("test_clipped.mp4", fourcc, 10.0, (frame_uint8_clipped.shape[1], frame_uint8_clipped.shape[0]))
out2.write(cv2.cvtColor(frame_uint8_clipped, cv2.COLOR_RGB2BGR))
out2.release()
print("Saved test_clipped.mp4")

print("\n✅ Test complete! Check the generated PNG and MP4 files.")
