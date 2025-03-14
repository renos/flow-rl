from jax import numpy as jnp, lax, vmap
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax
import jax

class ActorCritic(nn.Module):
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

from jax import numpy as jnp, random, jit, vmap
import numpy as np
import time
import distrax

# Set up the random key
key = random.PRNGKey(42)

# Model configuration - simplified
action_dim = 5  # Number of possible actions
layer_width = 64  # Width of hidden layers
num_layers = 4  # Total depth of networks
num_tasks = 20  # Number of tasks

# Data shapes
batch_size = 16
obs_dim = 8  # Adjust according to your observation space

# Create network
network = ActorCritic(
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
for _ in range(runs):
    pi, values = jitted_forward(network_params, test_observations, test_task_ids)
avg_inference_time = (time.time() - start) / runs

print(f"Average inference time over {runs} runs: {avg_inference_time*1000:.4f} ms per batch")

# Sample actions from the policy
key = random.PRNGKey(0)
actions = pi.sample(seed=key)
print(f"Sampled actions (first 5): {actions[:5]}")