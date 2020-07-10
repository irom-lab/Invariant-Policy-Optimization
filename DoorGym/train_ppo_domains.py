import os
import sys
import time
import pickle
from collections import deque
import gym
import numpy as np

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from trained_visionmodel.visionmodel import VisionModelXYZ
from util import add_noise, load_visionmodel, prepare_trainer, prepare_env

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_domains
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import IPython as ipy

import doorenv

def main(raw_args=None):
    
    print("Training with PPO.")

    # If this is being called as a function from another python script
    if raw_args is not None:
        args = get_args(raw_args)
    else:
        args = main_args

    if args.algo != 'ppo':
        raise NotImplementedError

    # Total number of envs (both domains)
    args.num_processes = args.num_envs1 + args.num_envs2

    knob_noisy = args.knob_noisy
    pretrained_policy_load = args.pretrained_policy_load

    args.world_path_domain1 = os.path.expanduser(args.world_path_domain1)
    args.world_path_domain2 = os.path.expanduser(args.world_path_domain2)

    # Env kwargs for domain 1
    env_kwargs1 = dict(port = args.port,
                    visionnet_input = args.visionnet_input,
                    unity = args.unity,
                    world_path = args.world_path_domain1)

    # Env kwargs for domain 2
    env_kwargs2 = dict(port = args.port,
                    visionnet_input = args.visionnet_input,
                    unity = args.unity,
                    world_path = args.world_path_domain2)


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    summary_name = args.log_dir + '{0}_{1}'
    writer = SummaryWriter(summary_name.format(args.env_name, args.save_name))

    # Make vector env for two domains (envs are concatenated)
    envs = make_vec_envs_domains(args.env_name,
                         args.seed,
                         args.num_processes,
                         args.gamma, 
                         args.log_dir, 
                         device, 
                         False, 
                         args.num_envs1,
                         args.num_envs2,
                         env_kwargs1=env_kwargs1,
                         env_kwargs2=env_kwargs2)

    # agly ways to access to the environment attirubutes
    if args.env_name.find('doorenv')>-1:
        if args.num_processes>1:
            visionnet_input = envs.venv.venv.visionnet_input
            nn = envs.venv.venv.nn
            env_name = envs.venv.venv.xml_path
        else:
            visionnet_input = envs.venv.venv.envs[0].env.env.env.visionnet_input
            nn = envs.venv.venv.envs[0].env.env.env.nn
            env_name = envs.venv.venv.envs[0].env.env.env.xml_path
        dummy_obs = np.zeros(nn*2+3)
    else:
        dummy_obs = envs.observation_space
        visionnet_input = None
        nn = None

    if pretrained_policy_load:
        print("loading", pretrained_policy_load)
        actor_critic, ob_rms = torch.load(pretrained_policy_load)
    else:
        actor_critic = Policy(
            dummy_obs.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    
    if visionnet_input: 
            visionmodel = load_visionmodel(env_name, args.visionmodel_path, VisionModelXYZ())  
            actor_critic.visionmodel = visionmodel.eval()
    actor_critic.nn = nn
    actor_critic.to(device)

    #disable normalizer
    vec_norm = get_vec_normalize(envs)
    vec_norm.eval()
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              dummy_obs.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    full_obs = envs.reset()
    initial_state = full_obs[:,:envs.action_space.shape[0]]

    if args.env_name.find('doorenv')>-1 and visionnet_input:
        obs = actor_critic.obs2inputs(full_obs, 0)
    else:
        if knob_noisy:
            obs = add_noise(full_obs, 0)
        else:
            obs = full_obs

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    best_training_reward = -np.inf

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        pos_control = False
        total_switches = 0
        prev_selection = ""
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                next_action = action 

            if pos_control:
                frame_skip = 2
                if step%(512/frame_skip-1)==0: current_state = initial_state
                next_action = current_state + next_action
                for kk in range(frame_skip):
                    full_obs, reward, done, infos = envs.step(next_action)
                    
                current_state = full_obs[:,:envs.action_space.shape[0]]
            else:
                full_obs, reward, done, infos = envs.step(next_action)

            # convert img to obs if door_env and using visionnet 
            if args.env_name.find('doorenv')>-1 and visionnet_input:
                obs = actor_critic.obs2inputs(full_obs, j)
            else:
                if knob_noisy:
                    obs = add_noise(full_obs, j)
                else:
                    obs = full_obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        writer.add_scalar("Value loss", value_loss, j)
        writer.add_scalar("action loss", action_loss, j)
        writer.add_scalar("dist entropy loss", dist_entropy, j)
        writer.add_scalar("Episode rewards", np.mean(episode_rewards), j)

        if np.mean(episode_rewards) > best_training_reward:
            best_training_reward = np.mean(episode_rewards)
            current_is_best = True
        else:
            current_is_best = False

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1 or current_is_best) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, args.env_name + "_{}.{}.pt".format(args.save_name,j)))

            if current_is_best:
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, args.env_name + "_{}.best.pt".format(args.save_name)))
   
 
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        DR=False # True #Domain Randomization
        ################## for multiprocess world change ######################
        if DR:
            print("changing world")

            envs.close_extras()
            envs.close()
            del envs

            envs = make_vec_envs_domains(args.env_name,
                         args.seed,
                         args.num_processes,
                         args.gamma, 
                         args.log_dir, 
                         device, 
                         False, 
                         env_kwargs1=env_kwargs1,
                         env_kwargs2=env_kwargs2)

            full_obs = envs.reset()
            if args.env_name.find('doorenv')>-1 and visionnet_input:
                obs = actor_critic.obs2inputs(full_obs, j)
            else:
                obs = full_obs
        #######################################################################

if __name__ == "__main__":
    main_args = get_args()

    # knob_noisy = args.knob_noisy
    # pretrained_policy_load = args.pretrained_policy_load

    # # Env kwargs for domain 1
    # env_kwargs1 = dict(port = args.port,
    #                 visionnet_input = args.visionnet_input,
    #                 unity = args.unity,
    #                 world_path = args.world_path_domain1)

    # # Env kwargs for domain 2
    # env_kwargs2 = dict(port = args.port,
    #                 visionnet_input = args.visionnet_input,
    #                 unity = args.unity,
    #                 world_path = args.world_path_domain2)

    # Run
    main()


