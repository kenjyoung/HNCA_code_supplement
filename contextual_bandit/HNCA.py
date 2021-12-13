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

def binary_log_counterfactual_probs(o, i, fr_logit, theta, centered):
	i = jnp.expand_dims(i,axis=1)
	t = jnp.expand_dims(theta,axis=0)

	#difference resulting from flipping each input
	if(centered):
		cf_diff = t*(-2*i)
	else:
		cf_diff = t*(1-2*i)
	#true fire rate
	log_fr = jx.nn.log_sigmoid(fr_logit)
	#one minus true fire rate
	log_neg_fr = jx.nn.log_sigmoid(-fr_logit)
	#fire rate with each input inverted individually
	log_fr_inverse = jx.nn.log_sigmoid(jnp.expand_dims(fr_logit, axis=2)+cf_diff)
	#one minus fire rate with each input inverted individually
	log_neg_fr_inverse = jx.nn.log_sigmoid(-(jnp.expand_dims(fr_logit, axis=2)+cf_diff))
	#true probability of output
	log_p = jnp.expand_dims(log_fr*o+log_neg_fr*(1-o),axis=2)
	#probability of output with each input inverted individually
	log_p_inverse = log_fr_inverse*jnp.expand_dims(o,axis=2)+log_neg_fr_inverse*(1-jnp.expand_dims(o,axis=2))
	i = i*jnp.ones(log_p_inverse.shape)
	#probability of output with each input set to zero individually
	log_p_0 = jnp.where(i==(-1 if centered else 0),log_p,log_p_inverse)
	#probability of output with each input set to one individually
	log_p_1 = jnp.where(i==1,log_p,log_p_inverse)
	return log_p_0, log_p_1
jit_binary_log_counterfactual_probs = jit(binary_log_counterfactual_probs, static_argnums=(4,))

def binary_grads(fr, i):
	#gradient of fire-rate
	theta_grads = jnp.expand_dims((1-fr)*fr,axis=2)*jnp.expand_dims(i,axis=1)
	b_grads = (1-fr)*fr
	return theta_grads, b_grads
jit_binary_grads = jit(binary_grads)

def binary_forward(x, theta, b, key):
	fr_logit = jnp.einsum('kj,ij->ik', theta,x)+b
	fr = jnp.clip(jx.nn.sigmoid(fr_logit),min_denom,1-min_denom)
	key, subkey = jx.random.split(key)
	output = jx.random.bernoulli(key=subkey,p=fr)
	return output, key, fr_logit, fr
jit_binary_forward = jit(binary_forward)

class binary_hidden_layer():
	def __init__(self, key, num_units, input_length, lr, use_adam=False, centered=False):
		self.key, subkey = jx.random.split(key)

		self.theta = jx.random.uniform(key=subkey, shape=[num_units, input_length], minval=-jnp.sqrt(6/(input_length+num_units)),maxval=jnp.sqrt(6/(input_length+num_units)))
		self.b = jnp.zeros(num_units)

		self.centered = centered

		if(use_adam):
			self.theta_optimizer = adam_optimizer(lr)
			self.b_optimizer = adam_optimizer(lr)
		else:
			self.theta_optimizer = sgd_optimizer(lr)
			self.b_optimizer = sgd_optimizer(lr)

		self.last_input = None
		self.last_output = None
		self.last_fr = None
		self.last_fr_logit = None

	def forward(self, x):
		self.last_input = x
		self.last_output, self.key, self.last_fr_logit, self.last_fr = jit_binary_forward(x,self.theta, self.b, self.key)
		if(self.centered):
			return 2*(self.last_output-0.5)
		else:
			return self.last_output

	def log_counterfactual_probs(self):
		return jit_binary_log_counterfactual_probs(self.last_output,self.last_input, self.last_fr_logit,self.theta, self.centered)

	def grads(self):
		#gradient of the fire rate
		#(batch,output,input)
		return jit_binary_grads(self.last_fr, self.last_input)

	def update(self, theta_grad, b_grad):
		self.theta = self.theta_optimizer(self.theta, theta_grad)
		self.b = self.b_optimizer(self.b, b_grad)

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
	log_p_0 = jnp.where(i==(-1 if centered else 0),log_p,log_p_inverse)
	#probability of output with each input set to one individually
	log_p_1 = jnp.where(i==1,log_p,log_p_inverse)
	return log_p_0, log_p_1
jit_softmax_log_counterfactual_probs = jit(softmax_log_counterfactual_probs, static_argnums=(4,5))

def softmax_grads(p, i, o, num_outputs):
	#this computes the gradient of the log probability that a specific output o was selected
	theta_grads = jnp.einsum('ij,ik->ijk',jnp.eye(num_outputs)[o]-p,i)
	b_grads = jnp.eye(num_outputs)[o]-p
	return theta_grads, b_grads
jit_softmax_grads = jit(softmax_grads,static_argnums=(3,))

def softmax_forward(x, theta, b, key):
	key, subkey = jx.random.split(key)
	l = jnp.einsum('kj,ij->ik',theta,x)+b
	p = jx.nn.softmax(l)
	output = jx.random.categorical(subkey, l)
	return output, key, p, l
jit_softmax_forward = jit(softmax_forward)

class softmax_output_layer():
	def __init__(self, key, num_outputs, input_length, lr, use_adam=False, centered=False):
		self.key, subkey = jx.random.split(key)
		self.centered = centered

		self.num_outputs = num_outputs
		self.theta = jx.random.uniform(key=subkey, shape=[num_outputs, input_length], minval=-jnp.sqrt(6/(input_length+num_outputs)),maxval=jnp.sqrt(6/(input_length+num_outputs)))
		self.b = jnp.zeros(num_outputs)

		if(use_adam):
			self.theta_optimizer = adam_optimizer(lr)
			self.b_optimizer = adam_optimizer(lr)
		else:
			self.theta_optimizer = sgd_optimizer(lr)
			self.b_optimizer = sgd_optimizer(lr)

		self.last_input = None
		self.last_output = None
		self.last_output_logit = None
		self.last_p = None

	def forward(self,x):
		self.last_input = x
		self.last_output, self.key, self.last_p, self.last_output_logit = jit_softmax_forward(x, self.theta, self.b, self.key)
		return self.last_output

	def log_counterfactual_probs(self):
		return jit_softmax_log_counterfactual_probs(self.last_output,self.last_input,self.last_output_logit,self.theta, self.centered, self.last_input.shape[0])

	def grads(self):
		return jit_softmax_grads(self.last_p, self.last_input, self.last_output, self.num_outputs)

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
num_hidden = config["num_hidden"]
hidden_width = config["hidden_width"]
use_adam = config["use_adam"]
use_centered = config["use_centered"]
use_baseline = config["use_baseline"]
if(use_baseline):
	gamma = config["gamma"]
else:
	gamma = 0.0

units = [input_length]+[hidden_width]*num_hidden
layer_keys = jx.random.split(key, len(units)+1)
key, layer_keys = layer_keys[0], layer_keys[1:]
hidden_layers = [binary_hidden_layer(layer_keys[x], units[x+1],units[x],alpha, use_adam, centered=use_centered) for x in range(len(units)-1)]
output_layer = softmax_output_layer(layer_keys[-1],num_classes,units[-1],alpha, use_adam, centered=use_centered)

def dynamic_binarize(key,x):
	return jx.random.bernoulli(key,x).astype(float)

def compute_mean_gradient_variance(grad_vars):
	total_grad_variance = 0.0
	total_parameter_count = 0
	for v, q in zip(grad_vars,[output_layer]+list(reversed(hidden_layers))):
		theta_param_count = q.theta.size
		b_param_count = q.b.size
		total_grad_variance += v[0]*theta_param_count+v[1]*b_param_count
		total_parameter_count += theta_param_count+b_param_count
	return total_grad_variance/total_parameter_count

def compute_softmax_update_expanded(i, o, p, R, num_outputs):
	grads = jit_softmax_grads(p,i,o,num_outputs)
	grad_estimator = -jnp.expand_dims(R,axis=1)
	theta_grad = grads[0]*jnp.expand_dims(grad_estimator, axis=2)
	b_grad = grads[1]*grad_estimator
	theta_update = jnp.mean(theta_grad, axis=0)
	b_update = jnp.mean(b_grad, axis=0)
	return theta_update, b_update, theta_grad, b_grad
jit_compute_softmax_update_expanded=jit(compute_softmax_update_expanded, static_argnums=(4,))


def compute_softmax_update(l,R):
	return jit_compute_softmax_update_expanded(l.last_input, l.last_output, l.last_p, R, l.num_outputs)

def compute_credit_factor(fr, info_0, info_1):
	log_p_0 = jnp.sum(info_0,axis=1)
	log_p_1 = jnp.sum(info_1,axis=1)
	delta = log_p_0-log_p_1
	rho_1 = 1/(fr+(1-fr)*jnp.exp(delta))
	rho_0 = 1/(fr*jnp.exp(-delta)+(1-fr))
	return rho_1-rho_0
jit_compute_credit_factor = jit(compute_credit_factor)

def compute_binary_update_expanded(i, o, fr, info, R):
	grads = jit_binary_grads(fr,i)
	credit_factor = jit_compute_credit_factor(fr,info[0],info[1])
	grad_estimator = -jnp.expand_dims(R,axis=1)*credit_factor
	theta_grad = grads[0]*jnp.expand_dims(grad_estimator, axis=2)
	b_grad = grads[1]*grad_estimator
	theta_update = jnp.mean(theta_grad, axis=0)
	b_update = jnp.mean(b_grad, axis=0)
	return theta_update, b_update, theta_grad, b_grad
jit_compute_binary_update_expanded=jit(compute_binary_update_expanded)

def compute_binary_update(l,info,R):
	return jit_compute_binary_update_expanded(l.last_input, l.last_output, l.last_fr, info, R)

def update(y_hat,y,baseline):
	grad_vars = []
	R = (y_hat==y).astype(float)
	theta_update, b_update, theta_grad, b_grad = compute_softmax_update(output_layer,R-baseline)

	info = output_layer.log_counterfactual_probs()
	output_layer.update(theta_update, b_update)

	grad_vars_theta = jnp.mean(jnp.var(theta_grad,axis=0))
	grad_vars_b = jnp.mean(jnp.var(b_grad,axis=0))
	grad_vars+=[[grad_vars_theta,grad_vars_b]]

	for h in reversed(hidden_layers):
		theta_update, b_update, theta_grad, b_grad = compute_binary_update(h,info,R-baseline)

		if(h is not hidden_layers[0]): info = h.log_counterfactual_probs()

		h.update(theta_update, b_update)

		grad_vars_theta = jnp.mean(jnp.var(theta_grad,axis=0))
		grad_vars_b = jnp.mean(jnp.var(b_grad,axis=0))
		grad_vars+=[[grad_vars_theta,grad_vars_b]]
	return jnp.mean(R),grad_vars
		

def forward(x):
	for h in hidden_layers:
		x = h.forward(x)
	y_hat = output_layer.forward(x)
	return y_hat

num_epochs = config["num_epochs"]
test_accuracies = []
train_accuracies = []
grad_variances = []
mean_grad_variances = []
baseline = 0.0
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
		R, grad_vars = update(y_hat, y, baseline)

		#if baseline is not updated from 0.0 it's equivalent to not using it
		if(use_baseline):
			baseline = gamma*baseline+(1-gamma)*R
		total_mean_grad_vars += compute_mean_gradient_variance(grad_vars)
		total_R += R
		num_steps+=1
		if(not total_grad_vars):
			total_grad_vars=grad_vars
		else:
			for t,v in zip(total_grad_vars, grad_vars):
				t[0]+=v[0]
				t[1]+=v[1]
	train_set_accuracy = total_R/num_steps
	train_accuracies += [train_set_accuracy]

	epoch_grad_vars=[[t[0]/num_steps,t[1]/num_steps] for t in total_grad_vars]
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
		print("test set accuracy:" +str(test_set_accuracy))
		print("train set accuracy:" +str(train_set_accuracy)+"\n")
		print("Gradient Variance:")
		print("Mean: "+str(total_mean_grad_vars/num_steps))
		for i, t in enumerate(reversed(total_grad_vars)):
			print("layer_"+str(i)+" theta_grad variance: "+str(t[0]/num_steps))
			print("layer_"+str(i)+" b_grad variance: "+str(t[1]/num_steps))

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
		'layers' : hidden_layers+[output_layer],
	}, f)
