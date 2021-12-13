import jax as jx
import jax.numpy as jnp
from jax import jit
import pickle as pkl
from tqdm import tqdm
import argparse
import json
import time
#add parent dir to import path
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizers import adam_optimizer, sgd_optimizer

min_denom = 0.00001

def grads(fr, i):
	theta_grads = jnp.expand_dims((1-fr)*fr,axis=2)*jnp.expand_dims(i,axis=1)
	b_grads = (1-fr)*fr
	return theta_grads, b_grads
jit_grads = jit(grads)

def forward(x, theta, b, key, sample=True):
	fr = jnp.clip(jx.nn.sigmoid(jnp.einsum('kj,ij->ik', theta,x)+b),min_denom,1-min_denom)
	if(sample):
		key, subkey = jx.random.split(key)
		output = jx.random.bernoulli(key=subkey,p=fr)
		key, subkey = jx.random.split(key)
		output_2 = jx.random.bernoulli(key=subkey,p=fr)
	else:
		output = None
		output_2 = None
	return output, output_2, key, fr
jit_forward = jit(forward, static_argnums=(4))

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

	def grads(self):
		#gradient of the fire rate
		#(batch,output,input)
		return jit_prior_grads(self.last_fr)

	def update(self, b_grad):
		self.b = self.b_optimizer(self.b, b_grad)

class binary_hidden_layer():
	def __init__(self, key, num_units, input_length, lr, use_adam=False):
		self.key, subkey = jx.random.split(key)

		self.theta = jx.random.uniform(key=subkey, shape=[num_units, input_length], minval=-jnp.sqrt(6/(input_length+num_units)),maxval=jnp.sqrt(6/(input_length+num_units)))
		self.b = jnp.zeros(num_units)

		if(use_adam):
			self.theta_optimizer = adam_optimizer(lr)
			self.b_optimizer = adam_optimizer(lr)
		else:
			self.theta_optimizer = sgd_optimizer(lr)
			self.b_optimizer = sgd_optimizer(lr)

		self.last_input = None
		self.last_output = None
		self.last_output_2 = None
		self.last_fr = None
		self.last_u = None

	def forward(self, x, sample=True):
		self.last_input = x
		self.last_output, self.last_output_2, self.key, self.last_fr = jit_forward(x,self.theta, self.b, self.key, sample)
		if(sample):
			return self.last_output

	def grads(self):
		#gradient of the fire rate
		#(batch,output,input)
		return jit_grads(self.last_fr, self.last_input)

	def update(self, theta_grad, b_grad):
		self.theta = self.theta_optimizer(self.theta, theta_grad)
		self.b = self.b_optimizer(self.b, b_grad)


mnist_dir = "../data/MNIST/"
with open(os.path.join(os.path.dirname(__file__),mnist_dir+"full_data"), 'rb') as f:
	data = pkl.load(f)

train_labels = data['train_lbl']
test_labels = data['test_lbl']
train_images = data['train_img']
test_images = data['test_img']

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="REINFORCE_LOO_FR.out")
parser.add_argument("--model", "-m", type=str, default="REINFORCE_LOO_FR.model")
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)


with open(args.config, 'r') as f:
	config = json.load(f)

#define model
output_length = train_images.shape[1]
batch_size = config["batch_size"]
alpha = config["alpha"]
num_hidden = config["num_hidden"]
hidden_width = config["hidden_width"]
use_adam = config["use_adam"]

decoder_units = [hidden_width]*(num_hidden+1)+[output_length]
decoder_keys = jx.random.split(key, len(decoder_units))
key, decoder_keys = decoder_keys[0], decoder_keys[1:]

encoder_units = decoder_units[::-1]
encoder_keys = jx.random.split(key, len(encoder_units))
key, encoder_keys = encoder_keys[0], encoder_keys[1:]

decoder_layers = [binary_hidden_layer(decoder_keys[x],decoder_units[x+1],decoder_units[x],alpha,use_adam) for x in range(len(decoder_units)-1)]
encoder_layers = [binary_hidden_layer(encoder_keys[x],encoder_units[x+1],encoder_units[x],alpha,use_adam) for x in range(len(encoder_units)-1)]

key, subkey = jx.random.split(key)
prior = prior_layer(subkey,encoder_units[-1], alpha, use_adam)

def dynamic_binarize(key,x):
	return jx.random.bernoulli(key,x).astype(float)

def forward_encode(x):
	#input x is a training example
	for q, p in zip(encoder_layers, reversed(decoder_layers)):
		x = q.forward(x)
		p.forward(x, sample=False)
	prior.forward(batch_size, sample=False)
	return x

#this is used for computing reward for second sample
def forward_R(x,i):
	first = True
	R=jnp.zeros(batch_size)
	for q, p in zip(encoder_layers[i:], list(reversed(decoder_layers))[i:]):
		if(not first):
			x = q.forward(x)
			encoder_fr = q.last_fr
			encoder_H = -jnp.sum(encoder_fr*jnp.log(encoder_fr)+(1-encoder_fr)*jnp.log(1-encoder_fr), axis=1)
			R+=encoder_H
		else:
			first = False

		#this is a hacky way to avoid updating p_last_fr here
		_, _, _, decoder_fr = jit_forward(x,p.theta, p.b, p.key, False)

		decoder_p = decoder_fr*q.last_input+(1-decoder_fr)*(1-q.last_input)
		log_decoder_p = jnp.sum(jnp.log(decoder_p),axis=1)
		R += log_decoder_p
	prior.forward(batch_size, sample=False)
	prior_p = prior.last_fr*x+(1-prior.last_fr)*(1-x)
	log_prior_p = jnp.sum(jnp.log(prior_p),axis=1)
	R+=log_prior_p
	return R

def compute_updates_expanded(q_i, p_i, p_fr, q_fr, q_o, q_o2, R, R_2):
	#compute updates for decoder
	p_prob = p_fr*q_i+(1-p_fr)*(1-q_i)
	p_grads = jit_grads(p_fr,p_i)
	p_theta_update = -jnp.mean(p_grads[0]*jnp.expand_dims((2*q_i-1)/p_prob,axis=2), axis=0)
	p_b_update = -jnp.mean(p_grads[1]*(2*q_i-1)/p_prob, axis=0)

	#compute updates for encoder
	q_entropy_grad= -(jnp.log(q_fr)-jnp.log(1-q_fr))

	q_prob = q_fr*q_o+(1-q_fr)*(1-q_o)
	q_prob2 = q_fr*q_o2+(1-q_fr)*(1-q_o2)
	q_grad_estimator = -(q_entropy_grad+0.5*(jnp.expand_dims(R+jnp.sum(jnp.log(p_prob),axis=1)-R_2,axis=1)*(2*q_o-1)/q_prob+
											jnp.expand_dims(R_2-(R+jnp.sum(jnp.log(p_prob),axis=1)),axis=1)*(2*q_o2-1)/q_prob2))
	q_grads = jit_grads(q_fr,q_i)
	q_theta_grad = q_grads[0]*jnp.expand_dims(q_grad_estimator,axis=2)
	q_b_grad = q_grads[1]*q_grad_estimator
	q_theta_update = jnp.mean(q_theta_grad, axis=0)
	q_b_update = jnp.mean(q_b_grad, axis=0)

	#compute encoder entropy and decoder probability for use in reward
	q_H = -jnp.sum(q_fr*jnp.log(q_fr)+(1-q_fr)*jnp.log(1-q_fr), axis=1)
	log_p_prob = jnp.sum(jnp.log(p_prob),axis=1)

	return p_theta_update, p_b_update, q_theta_update, q_b_update, q_theta_grad, q_b_grad, q_H, log_p_prob
jit_compute_updates_expanded = jit(compute_updates_expanded)

def compute_updates(p, q, R, R_2):
	p_i = p.last_input
	q_i = q.last_input
	p_fr = p.last_fr
	q_fr = q.last_fr
	q_o = q.last_output
	q_o2 = q.last_output_2
	return jit_compute_updates_expanded(q_i, p_i, p_fr, q_fr, q_o, q_o2, R, R_2)

def compute_prior_update(p_fr, q_o):
	p_prob = p_fr*q_o+(1-p_fr)*(1-q_o)
	p_grad = jit_prior_grads(p_fr)
	p_b_grad = -p_grad*(2*q_o-1)/p_prob
	p_b_update = jnp.mean(p_b_grad,axis=0)
	log_p_prob = jnp.sum(jnp.log(p_prob),axis=1)
	return p_b_update, p_b_grad, log_p_prob
jit_compute_prior_update = jit(compute_prior_update)

def update():
	R=jnp.zeros(batch_size)
	grad_vars = []
	#compute gradients for prior layer
	p_b_update, p_b_grad, log_p_prob = jit_compute_prior_update(prior.last_fr, encoder_layers[-1].last_output)
	prior.update(p_b_update)
	R+=log_p_prob
	#compute gradients for the rest of the network
	for q, p in zip(reversed(encoder_layers), decoder_layers):
		R_2 = forward_R(q.last_output_2, encoder_layers.index(q))
		p_theta_update, p_b_update, q_theta_update, q_b_update, q_theta_grad, q_b_grad, q_H, log_p_prob = compute_updates(p, q, R, R_2)
		p.update(p_theta_update, p_b_update)
		q.update(q_theta_update, q_b_update)

		grad_vars_theta = jnp.mean(jnp.var(q_theta_grad,axis=0))
		grad_vars_b = jnp.mean(jnp.var(q_b_grad,axis=0))
		grad_vars+=[[grad_vars_theta,grad_vars_b]]

		#accumulate reward
		R+=q_H+log_p_prob
	#return the sampled ELBO, add the final encoder entropy and decoder log prob, which would otherwise be skipped
	return(jnp.mean(R), grad_vars)

def compute_mean_gradient_variance(grad_vars):
	total_grad_variance = 0.0
	total_parameter_count = 0
	for v, q in zip(grad_vars,reversed(encoder_layers)):
		theta_param_count = q.theta.size
		b_param_count = q.b.size
		total_grad_variance += v[0]*theta_param_count+v[1]*b_param_count
		total_parameter_count += theta_param_count+b_param_count
	return total_grad_variance/total_parameter_count


def forward_decode(x):
	for d in decoder_layers:
		x = d.forward(x)
	return x

num_epochs = config["num_epochs"]
ELBOs = []
grad_variances = []
mean_grad_variances = []
for epoch in range(num_epochs):
	total_elbo = 0
	total_grad_vars = None
	total_mean_grad_vars = 0.0
	num_steps = 0
	start = time.time()
	indices = jnp.arange(train_images.shape[0])
	key, subkey = jx.random.split(key)
	indices = jx.random.shuffle(subkey,indices)
	key, subkey = jx.random.split(key)
	binary_train_images = dynamic_binarize(subkey,train_images[indices])

	for i in tqdm(range(binary_train_images.shape[0]//batch_size), disable=not args.verbose):
		x = forward_encode(binary_train_images[i*batch_size:(i+1)*batch_size])
		ELBO,grad_vars = update()
		total_mean_grad_vars += compute_mean_gradient_variance(grad_vars)
		num_steps+=1
		total_elbo+=ELBO
		if(not total_grad_vars):
			total_grad_vars=grad_vars
		else:
			for t,v in zip(total_grad_vars, grad_vars):
				t[0]+=v[0]
				t[1]+=v[1]

	ELBOs+=[total_elbo/num_steps]
	epoch_grad_vars=[[t[0]/num_steps,t[1]/num_steps] for t in total_grad_vars]
	grad_variances+=[epoch_grad_vars]
	mean_grad_variances+=[total_mean_grad_vars/num_steps]

	if(args.verbose):
		tqdm.write("ELBO: "+str(total_elbo/num_steps))
		tqdm.write("Gradient Variance:")
		tqdm.write("Mean: "+str(total_mean_grad_vars/num_steps))
		for i, t in enumerate(reversed(total_grad_vars)):
			tqdm.write("layer_"+str(i)+" theta_grad variance: "+str(t[0]/num_steps))
			tqdm.write("layer_"+str(i)+" b_grad variance: "+str(t[1]/num_steps))
		tqdm.write("epoch: "+str(epoch))
		tqdm.write("epoch time: "+str(time.time()-start))

with open(args.output, 'wb') as f:
	pkl.dump({
		**config,
		**{
			'ELBOS': ELBOs,
			'mean_gradient_variances': mean_grad_variances,
			'grad_variances' : grad_variances
		}
	}, f)
with open(args.model,'wb') as f:
	pkl.dump({
		'decoder_layers' : decoder_layers,
		'encoder_layers' : encoder_layers,
		'prior_layer': prior
	}, f)
