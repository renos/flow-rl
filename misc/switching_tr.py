import jax
import jax.numpy as jnp
import triton
import triton.language as tl
import jax_triton as jt
from functools import partial
from typing import Tuple, Optional, Dict, List, Any


@triton.jit
def leaky_relu(x):
    """Leaky ReLU activation function."""
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    batch_idx,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    T,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bt,
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a: jnp.ndarray, b: jnp.ndarray, activation: str = "") -> jnp.ndarray:
    """
    Perform matrix multiplication C = A @ B using Triton kernels.

    Args:
        a: Input matrix A with shape (M, K)
        b: Input matrix B with shape (K, N)
        activation: Activation function to apply ("" or "leaky_relu")

    Returns:
        Output matrix C with shape (M, N)
    """
    # Extract matrix dimensions
    M, K = a.shape
    T, K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} and {b.shape}"

    # Prepare output shape - same precision as inputs
    out_shape = jax.ShapeDtypeStruct(shape=(M, N), dtype=a.dtype)

    # Fixed block sizes - these can be tuned for specific hardware
    # Using reasonable defaults that work well on most GPUs
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    # Calculate strides for the JAX arrays (row-major layout)
    a_stride_row = K  # Number of elements to skip to move down one row in A
    a_stride_col = 1  # Number of elements to skip to move right one column in A

    b_stride_batch = T * N  # Elements to skip between batches
    b_stride_row = N  # Elements to skip between rows within a batch
    b_stride_col = 1  # Elements to skip between columns within a row

    # For the output, similar logic applies
    c_stride_row = N  # Number of elements to skip to move down one row in C
    c_stride_col = 1  # Number of elements to skip to move right one column in C

    # Compute the grid size
    num_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n,)

    # Call the Triton kernel through jax_triton
    return jt.triton_call(
        a,
        b,  # matrices a and b
        batch_idx=0,
        # Matrix dimensions
        M=M,
        N=N,
        K=K,
        T=T,
        # Strides
        stride_am=a_stride_row,
        stride_ak=a_stride_col,  #
        stride_bt=b_stride_batch,
        stride_bk=b_stride_row,
        stride_bn=b_stride_col,  #
        stride_cm=c_stride_row,
        stride_cn=c_stride_col,
        # Kernel config
        kernel=matmul_kernel,
        out_shape=out_shape,
        grid=grid,
        # Constants
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation,
    )


# Specialized matmul with leaky ReLU activation
def matmul_leaky_relu(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Matrix multiplication with leaky ReLU activation."""
    return matmul(a, b, activation="leaky_relu")


# Create sample matrices
M, K, N = 1024, 1024, 1024
T = 20
a = jnp.ones((M, K), dtype=jnp.float16)
b = jnp.ones((T, K, N), dtype=jnp.float16)

# Run the matrix multiplication
c = matmul(a, b)
print(f"Output shape: {c.shape}")

# With activation
c_activated = matmul(a, b, activation="leaky_relu")
print(f"Activated output shape: {c_activated.shape}")

# JIT-compiled version
jit_matmul = jax.jit(matmul)
c_jit = jit_matmul(a, b)
import numpy as np

np.testing.assert_allclose(a @ b, c_jit, rtol=1e-5, atol=1e-5)
print(f"JIT output shape: {c_jit.shape}")
