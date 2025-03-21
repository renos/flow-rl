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


# def matmul_kernel(x_ref, y_ref, z_ref):
#     @pl.when(pl.program_id(2) == 0)
#     def _():
#         z_ref[...] = jnp.zeros_like(z_ref)

#     z_ref[...] += x_ref[...] @ y_ref[...]


# @partial(jax.jit, static_argnames=["block_shape"])
# def matmul(x: jax.Array, y: jax.Array, *, block_shape):

#     bm, bk, bn = block_shape
#     m, k = x.shape
#     _, n = y.shape
#     return pl.pallas_call(
#         matmul_kernel,
#         out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
#         in_specs=[
#             pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
#             pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
#         ],
#         out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
#         grid=(m // bm, n // bn, k // bk),
#     )(x, y)


m, k, n = 128, 8192, 128
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float32)
y = random.normal(k2, (k, n), dtype=jnp.float32)
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
jax_matmul = x @ y
numpy_matmul = x_np @ y_np
pallas_matmul = matmul(x, y, block_shape=(64, 64, 64))

# Convert all results to numpy for comparison
jax_matmul_np = np.array(jax_matmul)
pallas_matmul_np = np.array(pallas_matmul)


# Compute relative errors
print("\n--- Relative Errors ---")
print("JAX @ vs NumPy @:")
max_err, mean_err = compute_relative_error(jax_matmul_np, numpy_matmul)
print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

print("JAX @ vs Pallas matmul:")
max_err, mean_err = compute_relative_error(jax_matmul_np, pallas_matmul_np)
print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

print("Numpy @ vs Pallas matmul:")
max_err, mean_err = compute_relative_error(numpy_matmul, pallas_matmul_np)
print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")

# # Try different block shapes to see if they affect accuracy
# print("\n--- Testing Different Block Shapes ---")
# block_shapes = [(32, 32, 64), (64, 64, 128), (128, 128, 256)]
# for block_shape in block_shapes:
#     print(f"\nBlock shape: {block_shape}")
#     try:
#         pallas_matmul_alt = matmul(x, y, block_shape=block_shape)
#         pallas_matmul_alt_np = np.array(pallas_matmul_alt)
#         max_err, mean_err = compute_relative_error(jax_matmul_np, pallas_matmul_alt_np)
#         print(f"JAX @ vs Pallas matmul:")
#         print(f"  Max: {max_err:.6e}, Mean: {mean_err:.6e}")
#     except Exception as e:
#         print(f"Error with block shape {block_shape}: {str(e)}")

# # Check if there's a remainder in K dimension
# print(f"\nChecking K dimension handling:")
# print(f"k={k}, block_k=64, remainder={k % 64}")
# if k % 64 != 0:
#     print(
#         "There is a remainder in the K dimension - the modified kernel should now handle this correctly."
#     )

# # Find the indices of the maximum error elements
# max_error_indices = np.unravel_index(
#     np.argmax(np.abs(jax_matmul_np - pallas_matmul_np)), jax_matmul_np.shape
# )
# print(f"\nLocation of maximum error: {max_error_indices}")
# print(f"JAX value at this location: {jax_matmul_np[max_error_indices]}")
# print(f"Pallas value at this location: {pallas_matmul_np[max_error_indices]}")
# print(
#     f"Absolute difference: {abs(jax_matmul_np[max_error_indices] - pallas_matmul_np[max_error_indices])}"
# )
