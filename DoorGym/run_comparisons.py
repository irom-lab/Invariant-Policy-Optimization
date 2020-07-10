"""
Code for comparing IPO with PPO on DoorGym domains with differing friction.
"""

from train_ppo_domains import main as train_ppo
from train_ipo import main as train_ipo
# from enjoy import main as enjoy

import numpy as np
import IPython as ipy

# Setup
num_seeds = 1 # Number of different seeds to evaluate on
num_envs1 = 24 # Number of envs in domain 1
num_envs2 = 8  # Number of envs in domain 2

for seed in range(num_seeds):

	print(" ")
	print("seed: ", seed)

	# Train PPO
	train_ppo(['--seed', str(seed),'--env-name', 'doorenv-v0', '--algo', 'ppo', '--num-steps', str(4096), '--save-name', 'ppo-test-frictionloss'+str(seed), '--world-path-domain1', '~/Documents/Research/DoorGym/world_generator/world/pull_blue_floatinghooktrain1/', '--world-path-domain2', '~/Documents/Research/DoorGym/world_generator/world/pull_blue_floatinghooktrain2/', '--lr', str(1e-3), '--num-env-steps', str(25000000), '--num-envs1', str(num_envs1), '--num-envs2', str(num_envs2)])

	# Train IPO
	train_ipo(['--seed', str(seed),'--env-name', 'doorenv-v0', '--algo', 'ipo', '--num-steps', str(4096), '--save-name', 'ipo-test-frictionloss'+str(seed), '--world-path-domain1', '~/Documents/Research/DoorGym/world_generator/world/pull_blue_floatinghooktrain1/', '--world-path-domain2', '~/Documents/Research/DoorGym/world_generator/world/pull_blue_floatinghooktrain2/', '--lr', str(0.0005), '--num-env-steps', str(25000000), '--num-envs1', str(num_envs1), '--num-envs2', str(num_envs2)])
