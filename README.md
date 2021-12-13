# HNCA Code Supplement
This directory contains code to reproduce the experiments in the paper titled "Hindsight Network Credit Assignment: Efficient Credit Assignment in Networks of Discrete Stochastic Neurons". Experiments are implemented in python3 using the JAX library. The contextual bandit experiments are located in the directory "contextual_bandit" while hierarchical VAE experiments are in "generative_modeling".

# Preparing MNIST data
Before running any of the algorithms included in this supplement it is necessary to download and prepare the MNIST dataset. To do so create the directory "data/MNIST" in the root directory of this repo, and place the four dataset files available at http://yann.lecun.com/exdb/mnist/ within this directory. Once this is done run
```bash
process_mnist.py
```
The dataset should now be prepared for use by the scripts in this repo.

# Contextual Bandit Experiments
The contextual_bandit directory contains the code for HNCA and REINFORCE agents in the python files with matching names. An experiment can be run using a command like the following:
```bash
python3 HNCA.py -v -c HNCA.json -o HNCA.out -m HNCA.model -s 0
```
The -v flag specifies we wish to see verbose output which consists of a progress bar and final summary for each epoch. '-c HNCA.json' indicates that we wish to use the hyperparameters specified in the json file HNCA.json. The included HNCA.json file will reproduce the 3 hidden layer experiment presented in the paper. The -o flag specifies the directory to save the data, while -m specifies where to save final model parameters. The -s flag specifies the random seed.

Experiments using reinforce and each estimator with the addition of a running average baseline can be run analogously:
```bash
python3 HNCA.py -v -c HNCA_with_baseline.json -o HNCA_with_baseline.out -m HNCA_with_baseline.model -s 0
python3 REINFORCE.py -v -c REINFORCE.json -o REINFORCE.out -m REINFORCE.model -s 0
python3 REINFORCE.py -v -c REINFORCE_with_baseline.json -o REINFORCE.out -m REINFORCE.model -s 0
```

# Hierarchical VAE Experiments
The generative_model directory contains the code for HNCA, REINFORCE, disARM, REINFORCE_LOO_FR and REINFORCE_LOO_IS in the python files with matching names. An experiment can be run using a command like the following:
```bash
python3 HNCA.py -v -c HNCA.json -o HNCA.out -m HNCA.model -s 0
```
The flags are largely the same as the contextual bandit experiments. The included config.json file will reproduce the experiment from the 3 layer experiment presented in the paper, note that num_hidden is set to 2 in this case because hidden layers are taken to mean non-output layers for encoder and decoder, thus 2 hidden layers means 3 layers total as in the following network where X represents the space of input pixels: X->q_1->q_2->q_3->p_3->p_2->p_1->X.

Analogous experiments with other gradients estimators can be run with
```bash
python3 HNCA.py -v -c HNCA_with_baseline.json -o HNCA_with_baseline.out -m HNCA_with_baseline.model -s 0
python3 disARM.py -v -c disARM.json -o disARM.out -m disARM.model -s 0
python3 REINFORCE.py -v -c REINFORCE.json -o REINFORCE.out -m REINFORCE.model -s 0
python3 REINFORCE.py -v -c REINFORCE_with_baseline.json -o REINFORCE_with_baseline.out -m REINFORCE_with_baseline.model -s 0
python3 REINFORCE_LOO.py -v -c REINFORCE_LOO.json -o REINFORCE_LOO.out -m REINFORCE_LOO.model -s 0
```

Ablation experiments for HNCA can be reproduced by modifying the "all_child" and "full_reward" values in the HNCA_with_baseline.json appropriately.

# Nonlinear Contextual Bandit Experiments
The nonlinear_contextual_bandit directory contains the code for HNCA and REINFORCE agents in the python files with matching names. The architecture, in this case, is a deterministic convolutional network followed by a single Bernoulli hidden layer and softmax output. An experiment can be run using a command like the following:
```bash
python3 HNCA.py -v -c HNCA.json -o HNCA.out -m HNCA.model -s 0
```
The included config.json file will reproduce the nonlinear contextual bandit experiment presented in the paper.

An experiment using reinforce can be run analogously:
```bash
python3 REINFORCE.py -v -c REINFORCE.json -o REINFORCE.out -m REINFORCE.model -s 0
```