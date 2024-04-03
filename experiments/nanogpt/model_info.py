"""Testing NanoGPT JAX model definition.

Inspired by: https://github.com/cgarciae/nanoGPT-jax/blob/master/train.py
"""

import jax
import jax.numpy as jnp
from model import GPT, GPTConfig

gpt2_tiny = GPTConfig(block_size=128, vocab_size=32000, n_layer=2, n_head=8, n_embd=512)
train_config = dict(
    learning_rate=0.001,
    weight_decay=0.1,
    beta1=1,
    beta2=1,
)

rng_key = jax.random.PRNGKey(0)
init_value = jnp.ones((1, 1), dtype=jnp.int32)

model = GPT(gpt2_tiny)
# initialize weights
# state = model.create_state(**train_config)
params = model.init(rng_key, init_value, train=False)
print("Model initialized...")

# Model description
print(model.tabulate(rng_key, init_value, train=False))
