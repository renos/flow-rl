import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
from typing import Tuple


def matmul_kernel(x_ref, A_ref, o_ref, block_x: int, block_a: int, block_d: int):
    row_id, col_id = pl.program_id(0), pl.program_id(1)
    col_slice = pl.dslice(col_id * block_d, block_d)
    A_mask_j = (col_id * block_d + jnp.arange(block_d) < A_ref.shape[1])[None, :]
    a_i = jnp.arange(block_a)
    x_mask_i = (row_id * block_x + jnp.arange(block_x) < x_ref.shape[0])[:, None]
    x_j = jnp.arange(block_a)

    def body_i(start_i, carry_i):
        o_prev = carry_i
        x_mask = x_mask_i & (start_i * block_a + x_j < x_ref.shape[1])[None, :]
        x = pl.load(
            x_ref,
            (pl.dslice(row_id * block_x, block_x), pl.dslice(start_i * block_a, block_a)),
            mask=x_mask,
        )
        a_mask = A_mask_j & (start_i * block_a + a_i < A_ref.shape[0])[:, None]
        a = pl.load(A_ref, (pl.dslice(start_i * block_a, block_a), col_slice), mask=a_mask)
        return pl.dot(x, a) + o_prev

    o_init = jnp.zeros((block_x, block_d), dtype=jnp.float32)
    o = lax.fori_loop(0, pl.cdiv(A_ref.shape[0], block_a), body_i, o_init)
    o_slice = (pl.dslice(row_id * block_x, block_x), pl.dslice(col_id * block_d, block_d))
    o_mask = (row_id * block_x + jnp.arange(block_x) < o_ref.shape[0])[:, None] & (
        col_id * block_d + jnp.arange(block_d) < o_ref.shape[1]
    )
    pl.store(o_ref, o_slice, o.astype(o_ref.dtype), mask=o_mask)


def pallas_matmul(x: jnp.ndarray, A: jnp.ndarray, block_x: int = 32, block_a: int = 32, block_d: int = 32) -> jnp.ndarray:
    """
    Performs matrix multiplication using Pallas.
    
    Args:
        x: Input matrix of shape (M, K)
        A: Weight matrix of shape (K, N)
        block_x: Block size for rows of output
        block_a: Block size for inner dimension
        block_d: Block size for columns of output
        
    Returns:
        Output matrix of shape (M, N)
    """
    M, K = x.shape
    K_A, N = A.shape
    
    # Validate shapes
    assert K == K_A, f"Inner dimensions must match: {K} != {K_A}"
    
    # Calculate grid dimensions
    grid_m = pl.cdiv(M, block_x)
    grid_n = pl.cdiv(N, block_d)
    
    # Define output shape
    output_shape = jax.ShapeDtypeStruct((M, N), x.dtype)
    
    # Launch the kernel
    return pl.pallas_call(
        matmul_kernel,
        out_shape=output_shape,
        grid=(grid_m, grid_n)
    )(x, A, block_x, block_a, block_d)


# Simple test for forward pass
def test_pallas_matmul():
    # Create a test case with matrices of size at least 16
    M, K, N = 32, 32, 32  # Using powers of 2 for better alignment
    
    # Create test matrices with predictable values
    x = jnp.ones((M, K), dtype=jnp.float32)
    x = x * jnp.arange(1, M*K+1, dtype=jnp.float32).reshape(M, K) * 0.01
    
    A = jnp.ones((K, N), dtype=jnp.float32)
    A = A * jnp.arange(1, K*N+1, dtype=jnp.float32).reshape(K, N) * 0.01
    
    # Expected result using JAX's native matmul
    expected = x @ A
    
    # Result using our Pallas implementation with blocks of at least 16
    print(f"Input dtypes: x={x.shape}, A={A.shape}")
    result = pallas_matmul(x, A, block_x=16, block_a=16, block_d=16)
    
    # Print inputs and outputs
    print("Input x shape:", x.shape)
    print("Input A shape:", A.shape)
    print("Output shape:", result.shape)
    
    # Compare results
    max_diff = jnp.max(jnp.abs(expected - result))
    print(f"\nMax difference: {max_diff}")
    
    return max_diff < 1e-5


if __name__ == "__main__":
    # Test the implementation
    success = test_pallas_matmul()
    print(f"\nTest {'passed' if success else 'failed'}")