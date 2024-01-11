import jax
import jax.numpy as jnp
from jax import random

import jax_scaled_arithmetics as jsa


def relu(input):
    zeros = jsa.as_scaled_array(jnp.zeros_like(input))
    return jsa.lax.scaled_max(zeros, input)


class MLP:
    def __init__(self, width=128, num_classes=10, num_layers=3, nonlin=relu):
        self.width = width
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.create_weights(random.PRNGKey(0))
        self.nonlin = nonlin

    def create_weights(self, key):
        key, self.fc1_weights = random.normal(key, (784, self.width))
        self.fc1_weights = jsa.as_scaled_array(self.fc1_weights)
        self.fc1_biases = jsa.as_scaled_array(jnp.zeros((self.width,)))

        key, self.fc2_weights = random.normal(key, (self.width, self.width))
        self.fc2_weights = jsa.as_scaled_array(self.fc2_weights)
        self.fc2_biases = jsa.as_scaled_array(jnp.zeros((self.width,)))

        key, self.fc3_weights = random.normal(key, (self.width, self.num_classes))
        self.fc3_weights = jsa.as_scaled_array(self.fc3_weights)
        self.fc3_biases = jsa.as_scaled_array(jnp.zeros((self.num_classes,)))
        return

    def forward(self, x):
        out = jsa.lax.scaled_dot_general(x, self.fc1_weights)
        out = self.nonlin(jsa.lax.scaled_add(out, self.fc1_biases))
        out = jsa.lax.scaled_dot_general(out, self.fc1_weights)
        out = self.nonlin(jsa.lax.scaled_add(out, self.fc1_biases))
        return jax.lax.scaled_add(jsa.lax.scaled_dot_general(out, self.fc3_weights), self.fc3_biases)
