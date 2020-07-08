"""
Code for comparing IPO with gradient descent on LQR problems with distractors.
"""

from run_distractor_lqr import main as distractor_lqr_GD
from run_ipo_lqr import main as distractor_lqr_IPO
from run_multilayer_lqr import main as distractor_lqr_multilayer
import numpy as np
import IPython as ipy

# Setup
num_seeds = 10 # Number of different seeds to evaluate on
num_domains = 2 # Number of domains to train on 
num_distractors = 1000 # Dimensionality of distractors

exp_cost_GD_logs = []
exp_cost_IPO_logs = []
exp_cost_multilayer_logs = []
exp_cost_LQR_logs = []

for seed in range(0, num_seeds):

	print(" ")
	print("seed: ", seed)

	# Run the different methods

	# Simple gradient descent
	exp_cost_GD, exp_cost_LQR = distractor_lqr_GD(['--seed', str(seed), '--num_domains', str(num_domains), '--verbose', str(0), '--num_distractors', str(num_distractors)])
	
	# Gradient descent with overparameterized class of policies
	exp_cost_multilayer, _ = distractor_lqr_multilayer(['--seed', str(seed), '--num_domains', str(num_domains), '--verbose', str(0), '--num_distractors', str(num_distractors)])
	
	# IPO
	exp_cost_IPO, _ = distractor_lqr_IPO(['--seed', str(seed), '--num_domains', str(num_domains), '--verbose', str(0), '--num_distractors', str(num_distractors)])


	exp_cost_GD_logs.append(exp_cost_GD)
	exp_cost_IPO_logs.append(exp_cost_IPO)
	exp_cost_multilayer_logs.append(exp_cost_multilayer)
	exp_cost_LQR_logs.append(exp_cost_LQR)


# Print mean and std. dev.
print(" ")
print("Exp. cost GD (mean): ", np.mean(exp_cost_GD_logs), ";", "std. dev.: ", np.std(exp_cost_GD_logs))
print("Exp. cost multi-layer (mean): ", np.mean(exp_cost_multilayer_logs), ";", "std. dev.: ", np.std(exp_cost_multilayer_logs))
print("Exp. cost IPO (mean): ", np.mean(exp_cost_IPO_logs), ";", "std. dev.: ", np.std(exp_cost_IPO_logs))
print("Exp. cost LQR oracle (mean): ", np.mean(exp_cost_LQR_logs), ";", "std. dev.: ", np.std(exp_cost_LQR_logs))


# Print results from all seeds (for debugging)
debug = False
if debug:
	print(" ")
	print("Exp. cost GD: ", exp_cost_GD_logs)
	print("Exp. cost multi-layer: ", exp_cost_multilayer_logs)
	print("Exp. cost IPO: ", exp_cost_IPO_logs)
	print("Exp. cost LQR oracle: ", exp_cost_LQR_logs)



