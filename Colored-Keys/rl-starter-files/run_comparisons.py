import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import numpy as np

import utils
import IPython as ipy
import csv

import subprocess
from scripts.multi_domain_train import main

# Environments to evaluate on (test domain)
env_eval = 'MiniGrid-ColoredKeysGrey-v0'

# Number of time-steps to run agent for per environment
episodes = 100

# Total number of evaluations (seeds)
num_evals = 2 # We used 10 in the paper

# Number of processes (environments to test on)
procs = 50 

# Action with highest probability is selected
argmax = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Initialize values to store
num_frames = []
returns_per_episode = []

# Number of time-steps to train for
num_frames_total = 120000

######################################
# Choose method to train and test
method = 'ipo' # 'ppo' or 'ipo'
######################################

for seed in range(num_evals): 

    print("Round: ", seed, "\n")

    # Train agents
    print("Training agent")

    if method == 'ppo':
        # Train PPO
        ipo_model = False
        model = 'MiniGrid-ColoredKeysPPO-' + str(seed)
        main(['--algo', 'ppo', '--domain1', 'MiniGrid-ColoredKeysRed-v0', '--domain2', 'MiniGrid-ColoredKeysGreen-v0', '--p1', '0.5', '--model', model, '--save-interval', '1', '--frames', str(num_frames_total), '--seed', str(seed), '--procs', str(48)]) 

    elif method == 'ipo':
        # Train using IPO
        model = 'MiniGrid-ColoredKeysIPO-' + str(seed)
        ipo_model = True
        main(['--algo', 'ipo', '--domain1', 'MiniGrid-ColoredKeysRed-v0', '--domain2', 'MiniGrid-ColoredKeysGreen-v0', '--p1', '0.5', '--model', model, '--save-interval', '1', '--frames', str(num_frames_total), '--seed', str(seed), '--procs', str(48), '--lr', str(0.0005)]) 
    else:
        raise ValueError("No such method supported")

    # Set seed for all randomness sources
    utils.seed(seed)

    # Load environments
    envs = []
    for i in range(procs):
        env = utils.make_env(env_eval, seed + 100 * i) # Different envs from training
        envs.append(env)

    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agents
    model_dir = utils.get_model_dir(model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir, ipo_model, device, argmax, procs)

    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent on test domain

    start_time = time.time()
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(procs, device=device)
    log_episode_num_frames = torch.zeros(procs, device=device)

    while log_done_counter < episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Record values of interest for comparison
    num_frames_seed = sum(logs["num_frames_per_episode"])
    return_per_episode_seed = utils.synthesize(logs["return_per_episode"])

    num_frames.append(num_frames_seed)
    returns_per_episode.append(return_per_episode_seed["mean"])

    # Clear envs
    env = None
    envs = None


# Print things
print("returns_per_episode (mean): ", np.mean(returns_per_episode))
print("num_frames (mean): ", np.mean(num_frames))


print(" ")
print("returns_per_episode (all seeds): ", returns_per_episode)
print("num_frames (all seeds): ", num_frames)
