import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax
import jax

class ActorCriticConvSymbolicCraftax(nn.Module):
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)
        # image_embedding = jnp.concatenate([image_embedding, obs[:, : CraftaxEnv.get_flat_map_obs_shape()]], axis=-1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb

from typing import Sequence, Callable, Optional, Any, Tuple, Union, Dict

# Import promote_dtype and default_kernel_init
from flax.linen.linear import default_kernel_init
from flax.linen.dtypes import promote_dtype
from flax.linen import Module
from jax import numpy as jnp, lax, vmap
from jax.random import PRNGKey
from flax.linen.initializers import constant, orthogonal, zeros

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
import math

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
            print(inputs.shape, task_idx.shape, kernels.shape, biases.shape)
            y = task_based_matmul_with_pallas(
                        inputs, task_idx, kernels, biases, self.use_bias
                    )
        return y

# Define the MoEDense layer with a JAX-compatible implementation
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
class ActorCriticMoE_nowork(nn.Module):
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
    


class ActorCriticMoE(nn.Module):
    action_dim: int
    layer_width: int
    num_layers: int  # Total depth of networks
    num_tasks: int   # Number of tasks
    activation: str = "tanh"

    def setup(self):
        # Initialize separate networks for each task
        self.actor_networks = [
            self.create_network(is_actor=True) 
            for _ in range(self.num_tasks)
        ]
        self.critic_networks = [
            self.create_network(is_actor=False) 
            for _ in range(self.num_tasks)
        ]

    def create_network(self, is_actor=True):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        # Define a simple MLP structure
        layers = []
        for _ in range(self.num_layers - 1):
            layers.append(
                nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )
            )
            layers.append(activation)
        # Output layer
        layers.append(
            nn.Dense(
                self.action_dim if is_actor else 1,
                kernel_init=orthogonal(0.01 if is_actor else 1.0),
                bias_init=constant(0.0),
            )
        )
        return nn.Sequential(layers)

    def __call__(self, x, task_ids):
        batch_size = x.shape[0]

        # Process input through all actor and critic networks, batching over networks
        actor_outputs = jnp.stack(
            [vmap(self.actor_networks[i])(x) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size, num_tasks, action_dim]
        
        critic_outputs = jnp.stack(
            [vmap(self.critic_networks[i])(x) for i in range(self.num_tasks)],
            axis=1,
        )  # Shape: [batch_size, num_tasks, 1]

        # Select the correct actor and critic output for each item in the batch based on task_ids
        task_ids_one_hot = jnp.eye(self.num_tasks)[task_ids]  # Shape: [batch_size, num_tasks]

        # Use the one-hot encoding to select the correct outputs
        selected_actor_outputs = jnp.sum(
            actor_outputs * task_ids_one_hot[:, :, None], axis=1
        )
        selected_critic_outputs = jnp.sum(
            critic_outputs * task_ids_one_hot[:, :, None], axis=1
        ).squeeze(-1)

        # Convert actor logits to distribution
        pi = distrax.Categorical(logits=selected_actor_outputs)

        return pi, selected_critic_outputs

from typing import Callable
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal, he_normal, normal, constant
import distrax

class DensePerTaskNoIndex(nn.Module):
    features: int
    num_tasks: int
    # Branch-free initializers
    kernel_init: Callable = lecun_normal()  # good default for linear layers
    bias_init:   Callable = constant(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray) -> jnp.ndarray:
        """x: [B, I], task_ids: [B] (ints)"""
        # Params have a leading task axis; rely on x.shape[-1] for in_features.
        # If this runs during init (outside jit), shapes are concrete.
        # If it *accidentally* runs under jit, lecun_normal doesn't branch on shape.
        kernel = self.param(
            "kernel", self.kernel_init, (self.num_tasks, x.shape[-1], self.features)
        )
        bias = self.param(
            "bias", self.bias_init, (self.num_tasks, self.features)
        )

        # One-hot gate (no indexing)
        gate = jax.nn.one_hot(task_ids, self.num_tasks, dtype=x.dtype)  # [B, T]

        # y[b,o] = sum_t gate[b,t] * sum_i x[b,i] * kernel[t,i,o] + gate[b,t]*bias[t,o]
        y = jnp.einsum("bt,bi,tio->bo", gate, x, kernel)
        y = y + jnp.einsum("bt,to->bo", gate, bias)
        return y


class MLPPerTaskNoIndex(nn.Module):
    hidden_width: int
    num_layers: int               # >= 1
    num_tasks: int
    out_features: int
    activation: str = "tanh"

    # Hidden: He/LeCun are fine; pick based on activation
    kernel_init_hidden: Callable = he_normal()     # good for ReLU; LeCun for tanh
    kernel_init_out:    Callable = normal(0.01)    # small logits / value head
    bias_init:          Callable = constant(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray) -> jnp.ndarray:
        act = nn.relu if self.activation == "relu" else nn.tanh
        hidden_init = he_normal() if self.activation == "relu" else lecun_normal()

        # First (or only) layer
        h = DensePerTaskNoIndex(
            features=self.hidden_width,
            num_tasks=self.num_tasks,
            kernel_init=hidden_init,
            bias_init=self.bias_init,
        )(x, task_ids)
        h = act(h)

        # Middle hidden layers (if any)
        for _ in range(self.num_layers - 2):
            h = DensePerTaskNoIndex(
                features=self.hidden_width,
                num_tasks=self.num_tasks,
                kernel_init=hidden_init,
                bias_init=self.bias_init,
            )(h, task_ids)
            h = act(h)

        # Output layer
        y = DensePerTaskNoIndex(
            features=self.out_features,
            num_tasks=self.num_tasks,
            kernel_init=self.kernel_init_out,
            bias_init=self.bias_init,
        )(h if self.num_layers > 1 else x, task_ids)
        return y


class ActorCriticMoE_single(nn.Module):
    action_dim: int
    layer_width: int
    num_layers: int   # >= 1
    num_tasks: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray):
        # Actor (small output scale for logits)
        actor_logits = MLPPerTaskNoIndex(
            hidden_width=self.layer_width,
            num_layers=self.num_layers,
            num_tasks=self.num_tasks,
            out_features=self.action_dim,
            activation=self.activation,
            kernel_init_out=normal(0.01),   # small for logits
        )(x, task_ids)

        # Critic (slightly larger OK)
        value = MLPPerTaskNoIndex(
            hidden_width=self.layer_width,
            num_layers=self.num_layers,
            num_tasks=self.num_tasks,
            out_features=1,
            activation=self.activation,
            kernel_init_out=lecun_normal(),
        )(x, task_ids).squeeze(-1)

        pi = distrax.Categorical(logits=actor_logits)
        return pi, value
