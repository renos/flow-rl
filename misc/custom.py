import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax import random
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax import lax
from jax.experimental.pallas import Slice


def matmul_kernel(x_ref, y_ref, o_ref, *, block_k):
    # acc = jnp.zeros((x_ref.shape[0], y_ref.shape[1]), jnp.float32)
    # num_blocks = x_ref.shape[1] // block_k
    # for k in range(num_blocks):
    #     x = x_ref[:, k * block_k : (k + 1) * block_k]
    #     y = y_ref[k * block_k : (k + 1) * block_k, :]
    #     acc += x @ y
    # o_ref[:, :] = acc.astype(o_ref.dtype)
    acc = jnp.zeros((x_ref.shape[0], y_ref.shape[1]), jnp.float32)
    num_blocks = x_ref.shape[1] // block_k

    def body_fun(k, acc):
        x_start_idx = k * block_k
        x_slice = x_ref[Slice(0, x_ref.shape[0]), Slice(x_start_idx, block_k)]

        y_start_idx = k * block_k
        y_slice = y_ref[Slice(y_start_idx, block_k), Slice(0, y_ref.shape[1])]

        return acc + x_slice @ y_slice

    acc = lax.fori_loop(0, num_blocks, body_fun, acc)
    o_ref[:, :] = acc.astype(o_ref.dtype)


@partial(jax.jit, static_argnames=["block_shape"])
def matmul(x, y, *, block_shape):
    block_m, block_n, block_k = block_shape
    # Ensure dimensions are compatible with blocks
    assert x.shape[0] % block_m == 0, "m must be divisible by block_m"
    assert y.shape[1] % block_n == 0, "n must be divisible by block_n"
    assert x.shape[1] == y.shape[0], "Inner dimensions must match"
    assert x.shape[1] % block_k == 0, "k must be divisible by block_k"

    fused_matmul = pl.pallas_call(
        partial(matmul_kernel, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(
            (x.shape[0], y.shape[1]),
            jnp.float32,
        ),
        in_specs=[
            pl.BlockSpec((block_m, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        grid=(x.shape[0] // block_m, y.shape[1] // block_n),
    )
    return fused_matmul(x, y)


m, k, n = 128, 8192, 128
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float16)
y = random.normal(k2, (k, n), dtype=jnp.float16)
x_np = np.array(x)
y_np = np.array(y)


def compute_relative_error(A, B):
    """Compute relative error between two matrices."""
    error = np.abs(A - B)
    norm_B = np.maximum(np.abs(B), 1e-10)  # Avoid division by zero
    relative_error = error / norm_B
    max_rel_error = np.max(relative_error)
    mean_rel_error = np.mean(relative_error)
    return max_rel_error, mean_rel_error


# print(ref_matmul_blocked(x, y))

# np.testing.assert_allclose(x @ y, x_np @ y_np, rtol=1e-5, atol=1e-5)


# np.testing.assert_allclose(
#     x @ y, matmul(x, y, block_shape=(64, 64, 64)), rtol=1e-5, atol=1e-5
# )

# Compute matrix multiplications
# jax_matmul = x @ y
# numpy_matmul = x_np @ y_np
# pallas_matmul = matmul(x, y, block_shape=(1, 64, 64))

# # Convert all results to numpy for comparison
# jax_matmul_np = np.array(jax_matmul)
# pallas_matmul_np = np.array(pallas_matmul)


# # Compute relative errors
# print("\n--- Relative Errors ---")
# print("JAX @ vs Pallas matmul:")
# max_err, mean_err = compute_relative_error(jax_matmul_np, pallas_matmul_np)
# print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

# print("Numpy @ vs Pallas matmul:")
# max_err, mean_err = compute_relative_error(numpy_matmul, pallas_matmul_np)
# print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

import time


def time_function(func, *args, n_runs=10, warmup=5, **kwargs):
    """
    Time a function execution with warmup runs.

    Args:
        func: Function to time
        *args: Arguments to pass to the function
        n_runs: Number of timing runs
        warmup: Number of warmup runs
        **kwargs: Keyword arguments to pass to the function

    Returns:
        List of execution times
    """
    # Warmup runs
    for _ in range(warmup):
        result = func(*args, **kwargs)
        # Ensure JAX computations are completed
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Actual timing runs
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        # Ensure JAX computations are completed
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        end_time = time.time()
        times.append(end_time - start_time)

    return times


# Set up matrices
m, k, n = 128, 8192, 128
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float16)
y = random.normal(k2, (k, n), dtype=jnp.float16)
x_np = np.array(x)
y_np = np.array(y)

print("Matrix shapes:")
print(f"x: {x.shape}, y: {y.shape}")


# Define functions to benchmark
@jax.jit
def jax_matmul_fn(x, y):
    return x @ y


def numpy_matmul_fn(x, y):
    return x @ y


def pallas_matmul_fn(x, y):
    return matmul(x, y, block_shape=(1, 64, 64))


# First, validate results
jax_result = jax_matmul_fn(x, y)
numpy_result = numpy_matmul_fn(x_np, y_np)
pallas_result = pallas_matmul_fn(x, y)

# Convert to numpy for comparison
jax_result_np = np.array(jax_result)
pallas_result_np = np.array(pallas_result)

print("\n--- Validation ---")
print("JAX @ vs Pallas matmul:")
max_err, mean_err = compute_relative_error(jax_result_np, pallas_result_np)
print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

print("Numpy @ vs Pallas matmul:")
max_err, mean_err = compute_relative_error(numpy_result, pallas_result_np)
print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

# Now run benchmarks
print("\n--- Benchmarking ---")
n_runs = 10

print("\nJAX Matrix Multiplication (x @ y):")
jax_times = time_function(jax_matmul_fn, x, y, n_runs=n_runs)
print(f"  Min: {min(jax_times)*1000:.4f} ms")
print(f"  Max: {max(jax_times)*1000:.4f} ms")
print(f"  Mean: {np.mean(jax_times)*1000:.4f} ms")
print(f"  Median: {np.median(jax_times)*1000:.4f} ms")

print("\nNumPy Matrix Multiplication (x_np @ y_np):")
numpy_times = time_function(numpy_matmul_fn, x_np, y_np, n_runs=n_runs)
print(f"  Min: {min(numpy_times)*1000:.4f} ms")
print(f"  Max: {max(numpy_times)*1000:.4f} ms")
print(f"  Mean: {np.mean(numpy_times)*1000:.4f} ms")
print(f"  Median: {np.median(numpy_times)*1000:.4f} ms")

print("\nPallas Matrix Multiplication:")
pallas_times = time_function(pallas_matmul_fn, x, y, n_runs=n_runs)
print(f"  Min: {min(pallas_times)*1000:.4f} ms")
print(f"  Max: {max(pallas_times)*1000:.4f} ms")
print(f"  Mean: {np.mean(pallas_times)*1000:.4f} ms")
print(f"  Median: {np.median(pallas_times)*1000:.4f} ms")

# Performance comparison
print("\n--- Performance Comparison ---")
jax_median = np.median(jax_times)
numpy_median = np.median(numpy_times)
pallas_median = np.median(pallas_times)

print(f"JAX vs NumPy: JAX is {numpy_median/jax_median:.2f}x faster than NumPy")
print(f"Pallas vs JAX: Pallas is {jax_median/pallas_median:.2f}x faster than JAX")
print(f"Pallas vs NumPy: Pallas is {numpy_median/pallas_median:.2f}x faster than NumPy")
