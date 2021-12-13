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

def log_counterfactual_probs(o, i, fr, fr_logit, theta):
	i = jnp.expand_dims(i,axis=1)
	t = jnp.expand_dims(theta,axis=0)
	#true fire rate
	log_fr = jx.nn.log_sigmoid(fr_logit)
	#one minus true fire rate
	log_neg_fr = jx.nn.log_sigmoid(-fr_logit)
	#fire rate with each input inverted individually
	log_fr_inverse = jx.nn.log_sigmoid(jnp.expand_dims(fr_logit, axis=2)+(t*(1-2*i)))
	#one minus fire rate with each input inverted individually
	log_neg_fr_inverse = jx.nn.log_sigmoid(-(jnp.expand_dims(fr_logit, axis=2)+(t*(1-2*i))))
	#true probability of output
	log_p = jnp.expand_dims(log_fr*o+log_neg_fr*(1-o),axis=2)
	#one minus probability of output
	log_neg_p = jnp.expand_dims(log_neg_fr*o+log_fr*(1-o),axis=2)
	#probability of output with each input inverted individually
	log_p_inverse = log_fr_inverse*jnp.expand_dims(o,axis=2)+log_neg_fr_inverse*(1-jnp.expand_dims(o,axis=2))
	#one minus probability of output with each input inverted individually
	log_neg_p_inverse = log_neg_fr_inverse*jnp.expand_dims(o,axis=2)+log_fr_inverse*(1-jnp.expand_dims(o,axis=2))
	i = i*jnp.ones(log_p_inverse.shape)
	#probability of output with each input set to zero individually
	log_p_0 = jnp.where(i==0,log_p,log_p_inverse)
	#one minus probability of output with each input set to zero individually
	log_neg_p_0 = jnp.where(i==0,log_neg_p,log_neg_p_inverse)
	#probability of output with each input set to one individually
	log_p_1 = jnp.where(i==1,log_p,log_p_inverse)
	#one minus probability of output with each input set to one individually
	log_neg_p_1 = jnp.where(i==1,log_neg_p,log_neg_p_inverse)
	return log_p_0, log_p_1, log_neg_p_0, log_neg_p_1
jit_log_counterfactual_probs = jit(log_counterfactual_probs)

def grads(fr, i):
	theta_grads = jnp.expand_dims((1-fr)*fr,axis=2)*jnp.expand_dims(i,axis=1)
	b_grads = (1-fr)*fr
	return theta_grads, b_grads
jit_grads = jit(grads)

def forward(x, theta, b, key, sample=True):
	fr_logit = jnp.einsum('kj,ij->ik', theta,x)+b
	fr = jnp.clip(jx.nn.sigmoid(fr_logit),min_denom,1-min_denom)
	if(sample):
		key, subkey = jx.random.split(key)
		output = jx.random.bernoulli(key=subkey,p=fr)
	else:
		output = None
	return output, key, fr_logit, fr
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
		self.last_fr = None
		self.last_fr_logit = None

	def forward(self, x, sample=True):
		self.last_input = x
		output, self.key, self.last_fr_logit, self.last_fr = jit_forward(x,self.theta, self.b, self.key, sample)
		if(sample):
			self.last_output = output
			return output

	def grads(self):
		#gradient of the fire rate
		#(batch,output,input)
		return jit_grads(self.last_fr, self.last_input)

	def log_counterfactual_probs(self, o=None):
		#(batch, output, input)
		#if no o is passed compute counterfactual probabilities of the last output, otherwise compute counterfactual probabilities for the passed x
		if(o is None):
			o = self.last_output
		return jit_log_counterfactual_probs(o,self.last_input,self.last_fr, self.last_fr_logit,self.theta)

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
output_length = train_images.shape[1]
batch_size = config["batch_size"]
alpha = config["alpha"]
num_hidden = config["num_hidden"]
hidden_width = config["hidden_width"]
use_adam = config["use_adam"]
all_child = config["all_child"]
use_baseline = config["use_baseline"]
full_reward = config["full_reward"]
if(use_baseline):
	gamma = config["gamma"]
else:
	gamma = 0.0

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

baselines = [0]*len(decoder_layers)

def dynamic_binarize(key,x):
	return jx.random.bernoulli(key,x).astype(float)

def compute_mean_gradient_variance(grad_vars):
	total_grad_variance = 0.0
	total_parameter_count = 0
	for v, q in zip(grad_vars,reversed(encoder_layers)):
		theta_param_count = q.theta.size
		b_param_count = q.b.size
		total_grad_variance += v[0]*theta_param_count+v[1]*b_param_count
		total_parameter_count += theta_param_count+b_param_count
	return total_grad_variance/total_parameter_count

def forward_encode(x):
	#input x is a training example
	for q, p in zip(encoder_layers, reversed(decoder_layers)):
		x = q.forward(x)
		p.forward(x, sample=False)
	prior.forward(batch_size, sample=False)
	return x

#This is only used for ablation experiment where full reward is used instead of foward reward
def previous_R():
	earlier_rewards=[]
	R=jnp.zeros(batch_size)
	for q, p in zip(encoder_layers, reversed(decoder_layers)):
		encoder_H = -jnp.sum(q.last_fr*jnp.log(q.last_fr)+(1-q.last_fr)*jnp.log(1-q.last_fr), axis=1)
		R+=encoder_H

		earlier_rewards+=[R]

		decoder_p = p.last_fr*q.last_input+(1-p.last_fr)*(1-q.last_input)
		log_decoder_p = jnp.sum(jnp.log(decoder_p),axis=1)

		R+=log_decoder_p
	return earlier_rewards

def compute_credit_factor(fr, q_info_0, q_info_1, neg_q_info_0, neg_q_info_1):
	log_p_0 = jnp.sum(q_info_0,axis=1)
	log_p_1 = jnp.sum(q_info_1,axis=1)
	delta = log_p_0-log_p_1
	rho_1 = jnp.where(delta<0,1/(fr+(1-fr)*jnp.exp(delta)),jnp.exp(-delta)/(fr*jnp.exp(-delta)+(1-fr)))
	rho_0 = jnp.where(delta>0,1/(fr*jnp.exp(-delta)+(1-fr)),jnp.exp(delta)/(fr+jnp.exp(delta)*(1-fr)))

	R_q_0 = -jnp.sum(jnp.exp(q_info_0)*q_info_0+jnp.exp(neg_q_info_0)*neg_q_info_0,axis=1)
	R_q_1 = -jnp.sum(jnp.exp(q_info_1)*q_info_1+jnp.exp(neg_q_info_1)*neg_q_info_1,axis=1)
	if(all_child):
		immediate_q_delta = rho_1*R_q_1-rho_0*R_q_0
	else:
		immediate_q_delta = R_q_1-R_q_0
	return rho_0, rho_1, immediate_q_delta
jit_compute_credit_factor = jit(compute_credit_factor)

def compute_updates_expanded(q_i, p_i, p_fr, p_fr_logit, p_theta, p_b, q_fr, q_o, last_q_info, last_p_fr, R, first):
	#compute updates for decoder
	p_prob = p_fr*q_i+(1-p_fr)*(1-q_i)
	last_p_prob = last_p_fr*q_o+(1-last_p_fr)*(1-q_o)
	p_grads = jit_grads(p_fr,p_i)
	p_theta_update = -jnp.mean(p_grads[0]*jnp.expand_dims((2*q_i-1)/p_prob,axis=2), axis=0)
	p_b_update = -jnp.mean(p_grads[1]*(2*q_i-1)/p_prob, axis=0)

	#compute updates for encoder
	immediate_p_0, immediate_p_1, _, _ = jit_log_counterfactual_probs(q_i, p_i, p_fr, p_fr_logit, p_theta)
	immediate_p_0 = jnp.sum(immediate_p_0, axis=1)
	immediate_p_1 = jnp.sum(immediate_p_1, axis=1)

	q_entropy_grad= -(jnp.log(q_fr)-jnp.log(1-q_fr))
	if(not first):
		rho_0, rho_1, immediate_q_delta = jit_compute_credit_factor(q_fr, last_q_info[0], last_q_info[1], last_q_info[2], last_q_info[3])
	else:
		rho_0 = rho_1 = 1.0
	if(all_child):
		immediate_p_delta = rho_1*immediate_p_1-rho_0*immediate_p_0
	else:
		immediate_p_delta = immediate_p_1-immediate_p_0

	q_grad_estimator = -(
						q_entropy_grad+
						immediate_p_delta+
						rho_1*jnp.log(last_p_fr)-rho_0*jnp.log(1-last_p_fr)+
						(rho_1-rho_0)*(jnp.sum(jnp.log(last_p_prob),axis=1, keepdims=True)-jnp.log(last_p_prob))+
						jnp.expand_dims(R,axis=1)*(rho_1-rho_0)+
						(0 if first else immediate_q_delta)
					)

	q_grads = jit_grads(q_fr,q_i)
	q_theta_grad = q_grads[0]*jnp.expand_dims(q_grad_estimator,axis=2)
	q_b_grad = q_grads[1]*q_grad_estimator
	q_theta_update = jnp.mean(q_theta_grad, axis=0)
	q_b_update = jnp.mean(q_b_grad, axis=0)

	#compute encoder entropy and decoder probability for use in reward
	q_H = -jnp.sum(q_fr*jnp.log(q_fr)+(1-q_fr)*jnp.log(1-q_fr), axis=1)
	log_p_prob = jnp.sum(jnp.log(p_prob),axis=1)

	return p_theta_update, p_b_update, q_theta_update, q_b_update, q_theta_grad, q_b_grad, q_H, log_p_prob
jit_compute_updates_expanded = jit(compute_updates_expanded,static_argnums=(11))

def compute_updates(p, q, last_q_info, last_p_fr, R, first):
	p_i = p.last_input
	q_i = q.last_input
	p_fr = p.last_fr
	q_fr = q.last_fr
	p_fr_logit = p.last_fr_logit
	p_theta = p.theta
	p_b = p.b
	q_o = q.last_output
	return jit_compute_updates_expanded(q_i, p_i, p_fr, p_fr_logit, p_theta, p_b, q_fr, q_o, last_q_info, last_p_fr, R, first)

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
	first = True
	last_q_info = None
	last_q_H = 0
	grad_vars = []
	#compute gradients for prior layer
	p_b_update, p_b_grad, log_p_prob = jit_compute_prior_update(prior.last_fr, encoder_layers[-1].last_output)
	prior.update(p_b_update)
	last_log_p_prob = log_p_prob
	last_p_fr = prior.last_fr
	#compute gradients for the rest of the network
	for i, (q, p) in enumerate(zip(reversed(encoder_layers), decoder_layers)):
		reward_signal = R

		if(full_reward):
			reward_signal += earlier_rewards[i]

		p_theta_update, p_b_update, q_theta_update, q_b_update, q_theta_grad, q_b_grad, q_H, log_p_prob = compute_updates(p, q, last_q_info, last_p_fr, reward_signal-baselines[i], first)

		if(q is not encoder_layers[0]):
			last_q_info = q.log_counterfactual_probs()

		p.update(p_theta_update, p_b_update)
		q.update(q_theta_update, q_b_update)

		grad_vars_theta = jnp.mean(jnp.var(q_theta_grad,axis=0))
		grad_vars_b = jnp.mean(jnp.var(q_b_grad,axis=0))
		grad_vars+=[[grad_vars_theta,grad_vars_b]]

		#if baseline is not updated from 0.0 it's equivalent to not using it
		if(use_baseline):
			if(all_child):
				baselines[i] = gamma*baselines[i]+(1-gamma)*jnp.mean(reward_signal+(0 if first else last_log_p_prob+last_q_H+log_p_prob))
			else:
				baselines[i] = gamma*baselines[i]+(1-gamma)*jnp.mean(reward_signal+(0 if first else last_log_p_prob))

		#accumulate reward
		R += last_log_p_prob+last_q_H

		#save decoder entropy and encoder log_prob to add to reward next time
		last_q_H = q_H
		last_p_fr = p.last_fr
		last_log_p_prob = log_p_prob
		first = False

	#return the sampled ELBO, add the final encoder entropy and decoder log prob, which would otherwise be skipped
	return(jnp.mean(R+last_q_H+last_log_p_prob), grad_vars)


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
		if(full_reward):
			earlier_rewards = previous_R()
			earlier_rewards.reverse()

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
		'prior_layer': prior,
		'baselines' : baselines
	}, f)
