import matplotlib.pyplot as plt
import pickle as pkl
import jax as jx
import jax.numpy as jnp
from jax import jit
import argparse
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizers import adam_optimizer, sgd_optimizer

min_denom = 0.00001

def forward_decode(x, decoder_layers):
	for d in decoder_layers:
		x = d.forward(x)
	return x

def forward(x, theta, b, key, sample=True):
	fr = jnp.clip(jx.nn.sigmoid(jnp.einsum('kj,ij->ik', theta,x)+b),min_denom,1-min_denom)
	if(sample):
		key, subkey = jx.random.split(key)
		output = jx.random.bernoulli(key=subkey,p=fr)
	else:
		output = None
	return output, key, fr
jit_forward = jit(forward, static_argnums=(4))

class binary_hidden_layer():
	def __init__(self, key, num_units, input_length, lr, use_adam=False):
		self.key, subkey = jx.random.split(key)

		self.theta = jx.random.uniform(key=subkey, shape=[num_units, input_length], minval=-jnp.sqrt(6/(input_length+num_units)),maxval=jnp.sqrt(6/(input_length+num_units)))
		self.b = jnp.zeros(num_units)

		#Gradients of log probability of selected action
		self.last_input = None
		self.last_output = None
		self.last_fr = None

	def forward(self, x, sample=True):
		self.last_input = x
		output, self.key, self.last_fr = jit_forward(x,self.theta, self.b, self.key, sample)
		if(sample):
			self.last_output = output
			return output

def prior_forward(b, key, batch_size, sample=True):
	fr = jnp.expand_dims(jnp.clip(jx.nn.sigmoid(b),min_denom,1-min_denom),axis=0)
	if(sample):
		key, subkey = jx.random.split(key)
		output = jx.random.bernoulli(key=subkey,p=fr, shape=[batch_size,fr.shape[1]])
	else: 
		output = None
	return output, key, fr
jit_prior_forward = jit(prior_forward,static_argnums=(2,3))

def prior_grads(fr):
	#gradient of fire-rate
	return (1-fr)*fr
jit_prior_grads = jit(prior_grads)

class prior_layer():
	def __init__(self, key, num_units, lr, use_adam=False):
		self.key, subkey = jx.random.split(key)
		self.b = jnp.zeros(num_units)

		if(use_adam):
			self.b_optimizer = adam_optimizer(lr)
		else:
			self.b_optimizer = sgd_optimizer(lr)

		self.last_output = None
		self.last_fr = None

	def forward(self, batch_size, sample=True):
		self.last_output, self.key, self.last_fr = jit_prior_forward(self.b, self.key, batch_size, sample)
		return self.last_output

	def grads(self):
		#This is now simply the gradient of the fire rate
		#(batch,output,input)
		return jit_prior_grads(self.last_fr)

	def update(self, b_grad):
		self.b = self.b_optimizer(self.b, b_grad)

parser = argparse.ArgumentParser()
parser.add_argument("--loadfile", "-l", type=str)
args = parser.parse_args()


with open(args.loadfile, 'rb') as f:
	data_dict = pkl.load(f)
	decoder_layers  = data_dict["decoder_layers"]
	prior = data_dict["prior_layer"]

w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 8
rows = 8
for i in range(1, columns*rows +1):
	x = prior.forward(1)[0]
	img = forward_decode(jnp.expand_dims(x,axis=0), decoder_layers)[0].reshape(28,28)
	fig.add_subplot(rows, columns, i)
	plt.imshow(img, cmap='gray')
plt.show()