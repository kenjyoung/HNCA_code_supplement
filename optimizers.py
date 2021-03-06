import jax.numpy as jnp
from jax import jit

def adam_update(a, b_1, b_2, eps, t, m, v, g, x):
	m = b_1*m+(1-b_1)*g
	v = b_2*v+(1-b_2)*g*g
	m_hat = m/(1-b_1**t)
	v_hat = v/(1-b_2**t)
	return (x-a*m_hat/(jnp.sqrt(v_hat)+eps), m, v)

jit_adam_update = jit(adam_update)

class adam_optimizer():
	def __init__(self, alpha, beta_1=0.9, beta_2=0.999, eps=1e-8):
		self.alpha = alpha
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.eps = eps

		self.t = 0
		self.m = 0
		self.v = 0

	def __call__(self, x, g):
		self.t += 1
		x, self.m, self.v = jit_adam_update(self.alpha, 
											self.beta_1, 
											self.beta_2, 
											self.eps, 
											self.t,
											self.m,
											self.v,
											g,
											x)

		return x

def sgd_update(a, x, g):
	return (x-a*g)

jit_sgd_update = jit(sgd_update)

class sgd_optimizer():
	def __init__(self, alpha):
		self.alpha = alpha

	def __call__(self, x, g):
		return jit_sgd_update(self.alpha, x, g)
