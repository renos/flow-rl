from jax import numpy as jnp, lax, vmap
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal, zeros
from typing import Sequence, Callable, Optional, Any, Tuple, Union, Dict
import functools

import distrax
import jax
from jax.random import PRNGKey

# Import promote_dtype and default_kernel_init
from flax.linen.linear import default_kernel_init
from flax.linen.dtypes import promote_dtype
from flax.linen import Module

# Define type aliases
Array = Any  # JAX array type
Shape = Tuple[int, ...]  # Shape type alias
Dtype = Any  # Dtype type alias
PrecisionLike = Any  # Precision type


import jax
import jax.numpy as jnp
from typing import Callable, Optional
from jax.random import PRNGKey
from flax import linen as nn
from flax.linen.initializers import zeros
import numpy as np

# Try to import Pallas
try:
    import jax.experimental.pallas as pl
    PALLAS_AVAILABLE = True
except ImportError:
    PALLAS_AVAILABLE = False

# Define task-based matrix multiplication using Pallas
def task_based_matmul_with_pallas(inputs, task_idx, kernels, biases, use_bias=True):
    """Memory-efficient task-based matrix multiplication using Pallas."""
    if not PALLAS_AVAILABLE:
        raise ImportError("Pallas is not available. Install it with pip install jax[experimental]")
    
    # Extract dimensions from the actual shapes
    batch_size, input_features = inputs.shape
    n_tasks, kernel_input_features, output_features = kernels.shape
    
    # Calculate padded dimensions to next power of 2 (at least 16)
    def next_power_of_2(n, min_size=16):
        p = 1
        while p < n or p < min_size:
            p *= 2
        return p
    
    padded_input_features = next_power_of_2(input_features)
    padded_output_features = next_power_of_2(output_features)
    
    # Pad inputs and kernels
    padded_inputs = jnp.pad(inputs, ((0, 0), (0, padded_input_features - input_features)))
    padded_kernels = jnp.pad(kernels, ((0, 0), 
                                      (0, padded_input_features - kernel_input_features),
                                      (0, padded_output_features - output_features)))
    
    # Pad biases if using them
    if use_bias:
        padded_biases = jnp.pad(biases, ((0, 0), (0, padded_output_features - output_features)))
    else:
        padded_biases = None
    
    # Define the Pallas kernel (without slices)
    def matmul_kernel(kernels_ref, biases_ref, inputs_ref, task_ref, output_ref):
        # Get the current program ID (batch index)
        batch_idx = pl.program_id(0)
        
        # Load the input vector for this batch
        x = pl.load(inputs_ref, (batch_idx, slice(None)))
        
        # Load the task index for this batch
        task = pl.load(task_ref, (batch_idx,))
        
        # Load the corresponding weight matrix for this task
        kernel = pl.load(kernels_ref, (task, slice(None), slice(None)))
        
        # Perform matrix multiplication
        x_2d = x.reshape((1, -1))
        result = pl.dot(x_2d, kernel).reshape(-1)
        
        # Add bias if needed
        if use_bias:
            bias = pl.load(biases_ref, (task, slice(None)))
            result = result + bias
        
        # Store the full result
        pl.store(output_ref, (batch_idx, slice(None)), result)
    
    # Prepare output shape for padded output
    padded_output_shape = jax.ShapeDtypeStruct((batch_size, padded_output_features), inputs.dtype)
    
    # Call the Pallas kernel (using the full padded shape)
    padded_result = pl.pallas_call(
        matmul_kernel,
        out_shape=padded_output_shape,
        grid=(batch_size,),
    )(padded_kernels, padded_biases if use_bias else None, padded_inputs, task_idx)
    
    # Extract only the relevant part of the output outside of Pallas
    return padded_result[:, :output_features]
import math

# def task_based_matmul_with_pallas(inputs, task_idx, kernels, biases, use_bias=True, max_grid_size=1024):
#     """Memory-efficient task-based matrix multiplication using Pallas with grid size limits.
    
#     Args:
#         inputs: Input tensor of shape [batch_size, input_features]
#         task_idx: Task indices of shape [batch_size]
#         kernels: Weight matrices of shape [n_tasks, input_features, output_features]
#         biases: Bias vectors of shape [n_tasks, output_features]
#         use_bias: Whether to use bias
#         max_grid_size: Maximum grid size to use (to avoid hardware limitations)
        
#     Returns:
#         Output tensor of shape [batch_size, output_features]
#     """
#     if not PALLAS_AVAILABLE:
#         raise ImportError("Pallas is not available. Install it with pip install jax[experimental]")
    
#     # Extract dimensions from the actual shapes
#     batch_size, input_features = inputs.shape
#     n_tasks, kernel_input_features, output_features = kernels.shape
    
#     # Calculate padded dimensions to next power of 2 (at least 16)
#     def next_power_of_2(n, min_size=16):
#         p = 1
#         while p < n or p < min_size:
#             p *= 2
#         return p
    
#     padded_input_features = next_power_of_2(input_features)
#     padded_output_features = next_power_of_2(output_features)
    
#     # Pad inputs and kernels
#     padded_inputs = jnp.pad(inputs, ((0, 0), (0, padded_input_features - input_features)))
#     padded_kernels = jnp.pad(kernels, ((0, 0), 
#                                       (0, padded_input_features - kernel_input_features),
#                                       (0, padded_output_features - output_features)))
    
#     # Pad biases if using them
#     if use_bias:
#         padded_biases = jnp.pad(biases, ((0, 0), (0, padded_output_features - output_features)))
#     else:
#         padded_biases = None
    
#     # Process in chunks to avoid grid size limitations
#     if batch_size > max_grid_size:
#         # Calculate number of chunks needed
#         num_chunks = math.ceil(batch_size / max_grid_size)
#         chunk_size = math.ceil(batch_size / num_chunks)
        
#         # Process each chunk separately and concatenate results
#         outputs = []
#         for i in range(num_chunks):
#             start_idx = i * chunk_size
#             end_idx = min((i + 1) * chunk_size, batch_size)
            
#             # Process this chunk
#             chunk_output = task_based_matmul_with_pallas(
#                 inputs[start_idx:end_idx],
#                 task_idx[start_idx:end_idx],
#                 kernels,
#                 biases,
#                 use_bias,
#                 max_grid_size
#             )
            
#             outputs.append(chunk_output)
        
#         # Concatenate results
#         return jnp.concatenate(outputs, axis=0)
    
#     # Define the Pallas kernel for direct per-example computation (no dynamic slicing)
#     def matmul_kernel(kernels_ref, biases_ref, inputs_ref, task_ref, output_ref):
#         # Get the current program ID (batch index)
#         batch_idx = pl.program_id(0)
        
#         # Load the input vector for this batch
#         x = pl.load(inputs_ref, (batch_idx, slice(None)))
        
#         # Load the task index for this batch
#         task = pl.load(task_ref, (batch_idx,))
        
#         # Load the corresponding weight matrix for this task
#         kernel = pl.load(kernels_ref, (task, slice(None), slice(None)))
        
#         # Perform matrix multiplication (reshaping to ensure proper dimensions)
#         x_2d = x.reshape((1, -1))
#         result = pl.dot(x_2d, kernel).reshape(-1)
        
#         # Add bias if needed
#         if use_bias:
#             bias = pl.load(biases_ref, (task, slice(None)))
#             result = result + bias
        
#         # Store the result
#         pl.store(output_ref, (batch_idx, slice(None)), result)
    
#     # Prepare output shape for padded output
#     padded_output_shape = jax.ShapeDtypeStruct((batch_size, padded_output_features), inputs.dtype)
    
#     # Call the Pallas kernel with appropriate grid size
#     padded_result = pl.pallas_call(
#         matmul_kernel,
#         out_shape=padded_output_shape,
#         grid=(batch_size,),
#     )(padded_kernels, padded_biases if use_bias else None, padded_inputs, task_idx)
    
#     # Extract only the relevant part of the output
#     return padded_result[:, :output_features]

# Default kernel initializer
def default_kernel_init(key, shape, dtype):
    stddev = 1.0 / jnp.sqrt(shape[1])
    return jax.random.normal(key, shape, dtype) * stddev

class MoEDense(nn.Module):
    """A Mixture of Experts linear transformation that maintains separate parameters for each task."""
    features: int
    n_tasks: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_pallas: bool = True  # Option to use Pallas for memory efficiency

    @nn.compact
    def __call__(self, inputs: Array, task_idx: int) -> Array:
        """Applies a task-specific linear transformation to the inputs.
        
        Args:
          inputs: The nd-array to be transformed.
          task_idx: Integer specifying which expert/task parameters to use.
        
        Returns:
          The transformed input.
        """
        input_features = inputs.shape[-1]
        
        # Create parameters for all tasks at once
        kernel_shape = (self.n_tasks, input_features, self.features)
        kernels = self.param('kernel', 
                           self.kernel_init,
                           kernel_shape,
                           self.param_dtype)
        
        if self.use_bias:
            bias_shape = (self.n_tasks, self.features)
            biases = self.param('bias', 
                              self.bias_init, 
                              bias_shape,
                              self.param_dtype)
        else:
            biases = None
            
        # Ensure task_idx is valid
        task_idx = jnp.clip(task_idx, 0, self.n_tasks - 1)
        
        # For scalar task_idx, direct indexing is most efficient
        if task_idx.ndim == 0:
            kernel = kernels[task_idx]
            bias = None if biases is None else biases[task_idx]
            
            # Apply the transformation
            y = jnp.dot(inputs, kernel)
            if bias is not None:
                y = y + bias
        else:
            y = task_based_matmul_with_pallas(
                        inputs, task_idx, kernels, biases, self.use_bias
                    )
        return y
        
# # Define the MoEDense layer with a JAX-compatible implementation
# class MoEDense(nn.Module):
#     """A Mixture of Experts linear transformation that maintains separate parameters for each task."""
#     features: int
#     n_tasks: int
#     use_bias: bool = True
#     dtype: Optional[Dtype] = None
#     param_dtype: Dtype = jnp.float32
#     precision: PrecisionLike = None
#     kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
#     bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

#     @nn.compact
#     def __call__(self, inputs: Array, task_idx: int) -> Array:
#         """Applies a task-specific linear transformation to the inputs.
        
#         Args:
#           inputs: The nd-array to be transformed.
#           task_idx: Integer specifying which expert/task parameters to use.
        
#         Returns:
#           The transformed input.
#         """
#         input_features = inputs.shape[-1]
#         #breakpoint()
        
#         # Create parameters for all tasks at once
#         kernel_shape = (self.n_tasks, input_features, self.features)
#         kernels = self.param('kernel', 
#                            self.kernel_init,
#                            kernel_shape,
#                            self.param_dtype)
        
#         if self.use_bias:
#             bias_shape = (self.n_tasks, self.features)
#             biases = self.param('bias', 
#                               self.bias_init, 
#                               bias_shape,
#                               self.param_dtype)
#         else:
#             biases = None
            
#         # Select the appropriate parameters using index_select
#         # def select_params(task_id):
#         #     # Ensure task_id is valid
#         #     task_id = jnp.clip(task_id, 0, self.n_tasks - 1)
#         #     kernel = jnp.take(kernels, task_id, axis=0)
#         #     #jax.debug.print('kernel shape: {}', kernel.shape)
#         #     bias = None if biases is None else jnp.take(biases, task_id, axis=0)
#         #     #jax.debug.print('bias shape: {}', bias.shape)
#         #     return kernel, bias
            
#         # # Use dynamic dispatch
#         # kernel, bias = select_params(task_idx)
#         # Ensure task_idx is valid
#         task_idx = jnp.clip(task_idx, 0, self.n_tasks - 1)
        
#         # One-hot approach for kernel selection
#         one_hot = jnp.eye(self.n_tasks, dtype=kernels.dtype)[task_idx]
        
#         # Memory-efficient implementation:
#         # 1. If task_idx is scalar, directly index into kernels
#         # 2. If task_idx is batched, use efficient broadcasting
#         if task_idx.ndim == 0:
#             # Scalar task_idx - directly index for maximum efficiency
#             kernel = kernels[task_idx]
#             bias = None if biases is None else biases[task_idx]
            
#             # Apply the transformation
#             y = jnp.dot(inputs, kernel)
#             if bias is not None:
#                 y = y + bias
#         else:
#             # Create selector tensor with one_hot encoding (batch_size, n_tasks, 1, 1)
#             selector = one_hot.reshape(one_hot.shape[0], one_hot.shape[1], 1, 1)
            
#             # Select kernels without materializing the full tensor
#             # This uses JAX's smart broadcasting to avoid memory explosion
#             # Shape: (batch_size, input_dim, output_dim)
#             kernel_selected = jnp.sum(kernels[None, :, :, :] * selector, axis=1)
            
#             # Apply the transformation with custom matmul to control memory usage
#             y = jax.vmap(jnp.dot)(inputs, kernel_selected)
#             jax.debug.breakpoint()
#             # Add bias if needed - with similar memory-efficient approach
#             if biases is not None:
#                 bias_selector = one_hot.reshape(one_hot.shape[0], one_hot.shape[1], 1)
#                 bias_selected = jnp.sum(biases[None, :, :] * bias_selector, axis=1)
#                 y = y + bias_selected
                
#         return y
        # one_hot = jnp.eye(self.n_tasks, dtype=kernels.dtype)[task_idx]

        # # Use einsum for batched matrix selection and multiplication
        # # Select the kernels for each batch item
        # kernel = jnp.einsum('bt,tkf->bkf', one_hot, kernels)
        # bias = None if biases is None else jnp.einsum('bt,tf->bf', one_hot, biases)
        
        # # Promote types for computation
        # inputs_t, kernel_t, bias_t = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        
        # # Perform the linear transformation
        # #jax.debug.print('inputs_t shape: {}', inputs_t.shape)
        # #jax.debug.print('kernel_t shape: {}', kernel_t.shape)
        # y = lax.dot_general(
        #     inputs_t, kernel_t, 
        #     (((inputs_t.ndim - 1,), (1,)), ((0), (0))),
        #     precision=self.precision
        # )
        
        # # Add bias if needed
        # #jax.debug.breakpoint()
        # if bias_t is not None:
        #     y += bias_t#jnp.reshape(bias_t, (1,) * (y.ndim - 1) + (-1,))
            
        # return y

# Redesigned ActorCritic using MoEDense
class ActorCriticMoE(nn.Module):
    action_dim: int
    layer_width: int
    num_layers: int  # Total depth of networks
    num_tasks: int   # Number of tasks
    activation: str = "tanh"

    def setup(self):
        # Choose activation function
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh
            
        # Actor network layers
        self.actor_hidden_layers = [
            MoEDense(
                features=self.layer_width,
                n_tasks=self.num_tasks,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            for _ in range(self.num_layers - 1)
        ]
        
        self.actor_output = MoEDense(
            features=self.action_dim,
            n_tasks=self.num_tasks,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )
        
        # Critic network layers
        self.critic_hidden_layers = [
            MoEDense(
                features=self.layer_width,
                n_tasks=self.num_tasks,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            for _ in range(self.num_layers - 1)
        ]
        
        self.critic_output = MoEDense(
            features=1,
            n_tasks=self.num_tasks,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )

    def get_actor_output(self, x, task_id):
        # Forward pass through actor network for a specific task
        for layer in self.actor_hidden_layers:
            x = layer(x, task_id)
            x = self.activation_fn(x)
        return self.actor_output(x, task_id)
    
    def get_critic_output(self, x, task_id):
        # Forward pass through critic network for a specific task
        for layer in self.critic_hidden_layers:
            x = layer(x, task_id)   
            x = self.activation_fn(x)
        return self.critic_output(x, task_id)
    
    def __call__(self, x, task_ids):
        # Use vmap to process the batch, applying the correct expert for each item based on task_id
        actor_logits = self.get_actor_output(x, task_ids)
        critic_values = self.get_critic_output(x, task_ids).squeeze(-1)
        
        # Convert actor logits to distribution
        pi = distrax.Categorical(logits=actor_logits)
        
        return pi, critic_values

# Test the implementation
from jax import numpy as jnp, random, jit, vmap
import numpy as np
import time

import jax
# from jax import profiler

# # 1. Create a trace directory to save profiling results

# trace_dir = "/home/renos/tensorboard"
# with profiler.trace(trace_dir, create_perfetto_link=True):

import pynvml
import time
import jax

# Initialize NVML
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU

print(f"Found {device_count} GPU(s)")
print(f"Using: {pynvml.nvmlDeviceGetName(handle)}")

if True:
    # Set up the random key
    key = random.PRNGKey(42)

    # Model configuration - simplified
    action_dim = 5  # Number of possible actions
    layer_width = 128  # Width of hidden layers
    num_layers = 4  # Total depth of networks
    num_tasks = 20  # Number of tasks

    # Data shapes
    batch_size = 8192
    obs_dim = 128  # Adjust according to your observation space

    # Create network
    network = ActorCriticMoE(
        action_dim=action_dim,
        layer_width=layer_width,
        num_layers=num_layers,
        num_tasks=num_tasks
    )

    # Initialize with correct shapes
    rng, _rng = random.split(key)
    init_x = jnp.zeros((1, obs_dim))  # Shape is (1, obs_dim) for a single example
    init_task_ids = jnp.zeros((1,), dtype=jnp.int32)  # Shape is (1,) for a single example
    network_params = network.init(_rng, init_x, init_task_ids)

    # Now create batched data for testing
    test_observations = random.normal(key, (batch_size, obs_dim))
    test_task_ids = random.randint(key, (batch_size,), 0, num_tasks)

    # Define the forward pass - we'll use apply directly with batched inputs
    def forward_pass(params, obs, task_ids):
        return network.apply(params, obs, task_ids)

    # JIT-compile for better performance
    jitted_forward = jit(forward_pass)

    # Run a test pass and measure time
    print("Running test inference...")
    start = time.time()
    pi, values = jitted_forward(network_params, test_observations, test_task_ids)
    inference_time = time.time() - start

    # Print results
    print(f"Inference time: {inference_time*1000:.4f} ms")
    print(f"Observations shape: {test_observations.shape}")
    print(f"Task IDs shape: {test_task_ids.shape}")
    print(f"Policy logits shape: {pi.logits.shape}")
    print(f"Values shape: {values.shape}")

    # Run multiple times to measure average performance
    print("\nRunning performance test...")
    runs = 10
    start = time.time()
    for i in range(runs):
        pi, values = jitted_forward(network_params, test_observations, test_task_ids)
        jax.block_until_ready(pi)
    avg_inference_time = (time.time() - start) / runs

    print(f"Average inference time over {runs} runs: {avg_inference_time*1000:.4f} ms per batch")

    # Sample actions from the policy
    key = random.PRNGKey(0)
    actions = pi.sample(seed=key)
    print(f"Sampled actions (first 5): {actions[:5]}")