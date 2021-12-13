import jax as jx
import jax.numpy as jnp
from jax import jit
import pickle as pkl
from tqdm import tqdm
import argparse
import json
import time

import haiku as hk

import os

from jax.experimental import optimizers

class nonlinear_network(hk.Module):
	def __init__(self, num_units, num_channels, num_conv):
		super().__init__(name=None)
		self.num_units = num_units
		self.num_channels = num_channels
		self.num_conv = num_conv

	def __call__(self, x):
		layers = []
		for i in range(self.num_conv):
			layers += [hk.Conv2D(self.num_channels, 3, padding='VALID'),jx.nn.relu]
		layers+=[
			hk.Flatten(),
			hk.Linear(self.num_units)
		]
		fr = hk.Sequential(layers)
		return fr(x)

def binary_forward(params, network, x, key):
	fr = jx.nn.sigmoid(network(params, x))
	key, subkey = jx.random.split(key)
	o = jx.random.bernoulli(key=subkey,p=fr)
	return o, key, fr

#note, we keep the loss seperated across samples to compute batchwise variance
def nonlinear_loss(params, network, x, o, reward, log_p_0, log_p_1):
	fr = jx.nn.sigmoid(network(params, x))
	delta = log_p_0-log_p_1
	rho_1 = 1/(fr+(1-fr)*jnp.exp(delta))
	rho_0 = 1/(fr*jnp.exp(-delta)+(1-fr))
	loss = -jnp.sum(fr*(rho_1-rho_0),axis=1)*reward
	return loss

class binary_network():
	def __init__(self, key, num_units, input_shape, num_channels, num_conv, lr, centered=False):
		self.t = 0
		self.centered = centered

		opt_init, self.opt_update, self.get_params = optimizers.adam(alpha)
		network = hk.without_apply_rng(hk.transform(lambda x: nonlinear_network(num_units, num_channels, num_conv)(x)))
		self.apply = network.apply
		self.key, subkey = jx.random.split(key)
		dummy_input = jnp.zeros(input_shape)
		params = network.init(subkey,dummy_input)

		self.opt_state = opt_init(params)

		self._forward = jit(binary_forward, static_argnums=(1,))
		self.opt_update = jit(self.opt_update)
		#need jacobian here to get batchwise grads
		self.loss_grad = jit(jx.jacrev(nonlinear_loss), static_argnums=(1,))

		self.last_input = None
		self.last_output = None
		self.last_fr_logit = None

	def forward(self, x):
		self.last_input = x
		self.last_output, self.key, self.last_fr_logit = self._forward(self.params(),self.apply, x, self.key)
		if(self.centered):
			return 2*(self.last_output-0.5)
		else:
			return self.last_output

	def backward(self, reward, log_p_0, log_p_1):
		#compute the gradient estimate for this network
		return self.loss_grad(self.params(), self.apply, self.last_input, self.last_output, reward, log_p_0, log_p_1)

	def params(self):
		return self.get_params(self.opt_state)

	def update(self, grads):
		self.opt_state = self.opt_update(self.t, grads, self.opt_state)

def softmax_log_counterfactual_probs(o, i, p_logit, theta, centered, batch_size):
	i = jnp.expand_dims(i,axis=1)
	t = jnp.expand_dims(theta,axis=0)
	#difference resulting from flipping each input
	if(centered):
		cf_diff = t*(-2*i)
	else:
		cf_diff = t*(1-2*i)
	#true probability of output
	log_p = jnp.expand_dims(jx.nn.log_softmax(p_logit)[jnp.arange(batch_size),o],axis=(1,2))
	#probability of output with each input inverted individually
	log_p_inverse = jnp.expand_dims(jx.nn.log_softmax(jnp.expand_dims(p_logit, axis=2)+cf_diff,axis=1)[jnp.arange(batch_size),o],axis=1)
	#probability of output with each input set to zero individually
	log_p_0 = jnp.sum(jnp.where(i==(-1 if centered else 0),log_p,log_p_inverse),axis=1)
	#probability of output with each input set to one individually
	log_p_1 = jnp.sum(jnp.where(i==1,log_p,log_p_inverse), axis=1)
	return log_p_0, log_p_1

def softmax_forward(params, network, x, key):
	key, subkey = jx.random.split(key)
	l = network(params, x)
	output = jx.random.categorical(subkey, l)
	return output, key, l

#note, we keep the loss seperated across samples to compute batchwise variance
def softmax_loss(params, network, x, output, reward):
	l = network(params, x)
	loss = -jx.nn.log_softmax(l)[jnp.arange(batch_size),output]*reward
	return loss

class softmax_output_layer():
	def __init__(self, key, num_outputs, input_length, alpha, centered=False):
		self.t = 0

		self.centered = centered

		opt_init, self.opt_update, self.get_params = optimizers.adam(alpha)
		output_layer = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(num_outputs)(x)))

		self.apply = output_layer.apply
		self.key, subkey = jx.random.split(key)
		dummy_states = jnp.zeros(input_length)
		params = output_layer.init(subkey,dummy_states)

		self.opt_state = opt_init(params)

		self._forward = jit(softmax_forward, static_argnums=(1,))
		self.opt_update = jit(self.opt_update)

		#need jacobian here to get batchwise grads
		self.loss_grad = jit(jx.jacrev(softmax_loss), static_argnums=(1,))
		self.log_cf_probs = jit(softmax_log_counterfactual_probs, static_argnums=(4,5))

		self.num_outputs = num_outputs

		self.opt_state = opt_init(params)

		self.last_input = None
		self.last_output = None
		self.last_logit = None

	def forward(self, x):
		self.last_input = x
		self.last_output, self.key, self.last_logit = self._forward(self.params(), self.apply, x, self.key)
		return self.last_output

	def backward(self, reward):
		g = self.loss_grad(self.params(), self.apply, self.last_input, self.last_output, reward)
		theta = jnp.transpose(self.params()['linear']['w'])
		log_p_0, log_p_1 = self.log_cf_probs(self.last_output, self.last_input, self.last_logit, theta, self.centered, batch_size)
		return g, log_p_0, log_p_1

	def params(self):
		return self.get_params(self.opt_state)

	def update(self, grads):
		self.opt_state = self.opt_update(self.t, grads, self.opt_state)

mnist_dir = "../data/MNIST/"
with open(os.path.join(os.path.dirname(__file__),mnist_dir+"full_data"), 'rb') as f:
	data = pkl.load(f)

train_labels = data['train_lbl']
test_labels = data['test_lbl']
train_images = data['train_img']
test_images = data['test_img']
train_images = jnp.reshape(train_images, (train_images.shape[0],28,28,1))
test_images = jnp.reshape(test_images, (test_images.shape[0],28,28,1))

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="HNCA.out")
parser.add_argument("--model", "-m", type=str, default="HNCA.model")
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)


with open(args.config, 'r') as f:
	config = json.load(f)

#define model
input_length = train_images.shape[1]
num_classes = 10
batch_size = config["batch_size"]
alpha = config["alpha"]
num_conv = config["num_conv"]
hidden_width = config["hidden_width"]
num_channels = config["num_channels"]
use_centered = config["use_centered"]

key, subkey = jx.random.split(key)
network = binary_network(subkey, hidden_width, (batch_size,28,28,1), num_channels, num_conv, alpha, use_centered)
key, subkey = jx.random.split(key)
output_layer = softmax_output_layer(subkey,num_classes,hidden_width,alpha, use_centered)

def dynamic_binarize(key,x):
	return jx.random.bernoulli(key,x).astype(float)

def compute_grad_var(grads):
	grad_vars = jx.tree_map(lambda x: jnp.mean(jnp.var(x,axis=0)), grads)
	total_grad_vars = jx.tree_map(lambda x: jnp.sum(jnp.var(x,axis=0)), grads)
	total_grad_vars = jx.tree_util.tree_reduce(lambda x, y: x+y, total_grad_vars, 0.0)
	total_params = jx.tree_map(lambda x: sum(x.shape), grads)
	total_params = jx.tree_util.tree_reduce(lambda x, y: x+y, total_params, 0.0)
	mean_grad_vars = total_grad_vars/total_params
	return grad_vars, mean_grad_vars
compute_grad_var = jit(compute_grad_var)

def forward(x):
	x = network.forward(x)
	y_hat = output_layer.forward(x)
	return y_hat

def tree_sum(t):
	return jx.tree_map(lambda x: jnp.sum(x, axis=0), t)
tree_sum = jit(tree_sum)

def tree_mean(t):
	return jx.tree_map(lambda x: jnp.mean(x, axis=0), t)
tree_mean = jit(tree_mean)

def tree_add(t,s):
	return jx.tree_multimap(lambda x,y: x+y, t, s)
tree_add = jit(tree_add)

def tree_div(t,d):
	return jx.tree_map(lambda x: x/d, t)
tree_div = jit(tree_div)

def backward(y_hat,y):
	R = (y_hat==y).astype(float)
	output_grads, log_p_0, log_p_1 = output_layer.backward(R)

	update = tree_mean(output_grads)
	output_layer.update(update)

	network_grads = network.backward(R, log_p_0, log_p_1)
	update = tree_mean(network_grads)
	network.update(update)

	all_grads = {**output_grads, **network_grads}
	grad_vars, mean_grad_vars = compute_grad_var(all_grads)
	return jnp.mean(R),grad_vars, mean_grad_vars

num_epochs = config["num_epochs"]
test_accuracies = []
train_accuracies = []
grad_variances = []
mean_grad_variances = []
for epoch in range(num_epochs):
	total_grad_vars = None
	total_mean_grad_vars = 0.0
	num_steps = 0
	start = time.time()
	indices = jnp.arange(train_images.shape[0])
	key, subkey = jx.random.split(key)
	indices = jx.random.shuffle(subkey,indices)
	key, subkey = jx.random.split(key)
	random_train_images = dynamic_binarize(subkey,train_images[indices])
	random_train_labels = train_labels[indices]

	total_R = 0
	for i in tqdm(range(random_train_images.shape[0]//batch_size), disable=not args.verbose):
		y_hat = forward(random_train_images[i*batch_size:(i+1)*batch_size])
		y = random_train_labels[i*batch_size:(i+1)*batch_size]

		R, grad_vars, mean_grad_vars = backward(y_hat, y)
		total_mean_grad_vars += mean_grad_vars
		total_R += R
		num_steps+=1
		if(not total_grad_vars):
			total_grad_vars = grad_vars
		else:
			total_grad_vars=tree_add(total_grad_vars, grad_vars)

	train_set_accuracy = total_R/num_steps
	train_accuracies += [train_set_accuracy]

	epoch_grad_vars=[tree_div(total_grad_vars,num_steps)]
	grad_variances+=[epoch_grad_vars]
	mean_grad_variances+=[total_mean_grad_vars/num_steps]

	total_R = 0
	num_steps = 0
	key, subkey = jx.random.split(key)
	binary_test_images = dynamic_binarize(subkey,test_images)
	for i in tqdm(range(binary_test_images.shape[0]//batch_size), disable=not args.verbose):
		y = test_labels[i*batch_size:(i+1)*batch_size]
		y_hat = forward(binary_test_images[i*batch_size:(i+1)*batch_size])
		total_R += jnp.mean(y==y_hat)
		num_steps+=1
	test_set_accuracy = total_R/num_steps
	test_accuracies += [test_set_accuracy]
	if(args.verbose):
		print("epoch: "+str(epoch))
		print("epoch time: "+str(time.time()-start))
		print("test set accuracy: " +str(test_set_accuracy))
		print("train set accuracy: " +str(train_set_accuracy)+"\n")
		print("Mean Gradient Variance: "+str(mean_grad_variances[-1]))

with open(args.output, 'wb') as f:
	pkl.dump({
		**config,
		**{
			'mean_gradient_variances': mean_grad_variances,
			'grad_variances' : grad_variances,
			'test_set_accuracies': test_accuracies,
			'train_set_accuracies': train_accuracies
		}
	}, f)
with open(args.model,'wb') as f:
	pkl.dump({
		'network_params' : network.params(),
		'output_params' : output_layer.params()
	}, f)
