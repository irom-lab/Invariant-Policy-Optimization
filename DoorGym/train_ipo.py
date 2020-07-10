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
from a2c_ppo_acktr.model import Policy_av, Policy
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

    # If this is being called as a function from another python script
    if raw_args is not None:
        args = get_args(raw_args)
    else:
        args = main_args

    if args.algo != 'ipo':
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


    print("Training with IPO.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    summary_name = args.log_dir + '{0}_{1}'
    writer = SummaryWriter(summary_name.format(args.env_name, args.save_name))

    # Make vector env for two domains (each contains num_processes/2 envs)
    envs1 = make_vec_envs(args.env_name,
                         args.seed,
                         args.num_envs1,
                         args.gamma, 
                         args.log_dir, 
                         device, 
                         False, 
                         env_kwargs=env_kwargs1)

    envs2 = make_vec_envs(args.env_name,
                         args.seed,
                         args.num_envs2,
                         args.gamma, 
                         args.log_dir, 
                         device, 
                         False, 
                         env_kwargs=env_kwargs2)


    # agly ways to access to the environment attirubutes
    if args.env_name.find('doorenv')>-1:
        visionnet_input = envs1.venv.venv.visionnet_input
        nn = envs1.venv.venv.nn
        env_name = envs1.venv.venv.xml_path
            
        dummy_obs = np.zeros(nn*2+3)
    else:
        dummy_obs = envs1.observation_space
        visionnet_input = None
        nn = None

    if pretrained_policy_load:
        print("loading", pretrained_policy_load)
        actor_critic, ob_rms = torch.load(pretrained_policy_load)
    else:
        actor_critic = Policy_av(
            dummy_obs.shape,
            envs1.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})

        # actor_critic = Policy(
        #     dummy_obs.shape,
        #     envs1.action_space,
        #     base_kwargs={'recurrent': args.recurrent_policy})
    
    if visionnet_input: 
        raise NotImplementedError
        visionmodel = load_visionmodel(env_name, args.visionmodel_path, VisionModelXYZ())  
        actor_critic.visionmodel = visionmodel.eval()

    actor_critic.nn = nn
    actor_critic.to(device)

    #disable normalizer
    vec_norm1 = get_vec_normalize(envs1)
    vec_norm1.eval()
    vec_norm2 = get_vec_normalize(envs2)
    vec_norm2.eval()
    
    # Create two agents (one for each domain)
    params1 = [{'params': actor_critic.base.actor1.parameters()}, 
    {'params': actor_critic.base.critic1.parameters()}, 
    {'params': actor_critic.base.critic_linear1.parameters()},
    {'params': actor_critic.base.fc_mean1.parameters()},
    {'params': actor_critic.base.logstd1.parameters()}]

    params2 = [{'params': actor_critic.base.actor2.parameters()}, 
    {'params': actor_critic.base.critic2.parameters()}, 
    {'params': actor_critic.base.critic_linear2.parameters()},
    {'params': actor_critic.base.fc_mean2.parameters()},
    {'params': actor_critic.base.logstd2.parameters()}]

    # params1 = None
    # params2 = None

    agent1 = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        optim_params = params1)

    agent2 = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        optim_params = params2)


    # Rollout storage for each domain
    rollouts1 = RolloutStorage(args.num_steps, args.num_envs1,
                              dummy_obs.shape, envs1.action_space,
                              actor_critic.recurrent_hidden_state_size)

    rollouts2 = RolloutStorage(args.num_steps, args.num_envs2,
                              dummy_obs.shape, envs2.action_space,
                              actor_critic.recurrent_hidden_state_size)


    full_obs1 = envs1.reset()
    initial_state1 = full_obs1[:,:envs1.action_space.shape[0]]

    full_obs2 = envs2.reset()
    initial_state2 = full_obs2[:,:envs2.action_space.shape[0]]

    if args.env_name.find('doorenv')>-1 and visionnet_input:
        obs1 = actor_critic.obs2inputs(full_obs1, 0)
        obs2 = actor_critic.obs2inputs(full_obs2, 0)
    else:
        if knob_noisy:
            obs1 = add_noise(full_obs1, 0)
            obs2 = add_noise(full_obs2, 0)
        else:
            obs1 = full_obs1
            obs2 = full_obs2

    rollouts1.obs[0].copy_(obs1)
    rollouts1.to(device)

    rollouts2.obs[0].copy_(obs2)
    rollouts2.to(device)

    episode_rewards1 = deque(maxlen=10)
    episode_rewards2 = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    num_updates = int(num_updates/2) # Since have two domains per iteration

    best_training_reward = -np.inf

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent1.optimizer, j, num_updates, args.lr)
            utils.update_linear_schedule(
                agent2.optimizer, j, num_updates, args.lr)

        ################## Do rollouts and updates for domain 1 ##################

        pos_control = False
        total_switches = 0
        prev_selection = ""
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts1.obs[step], rollouts1.recurrent_hidden_states[step],
                    rollouts1.masks[step])
                next_action = action 

            try:
                # print(next_action)
                full_obs, reward, done, infos = envs1.step(next_action)
            except:
                ipy.embed()

            if knob_noisy:
                obs = add_noise(full_obs, j)
            else:
                obs = full_obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards1.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts1.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts1.obs[-1], rollouts1.recurrent_hidden_states[-1],
                rollouts1.masks[-1]).detach()

        rollouts1.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent1.update(rollouts1)
        rollouts1.after_update()
        value_loss1 = value_loss
        action_loss1 = action_loss
        dist_entropy1 = dist_entropy

        ################## Do rollouts and updates for domain 2 ##################

        pos_control = False
        total_switches = 0
        prev_selection = ""
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts2.obs[step], rollouts2.recurrent_hidden_states[step],
                    rollouts2.masks[step])
                next_action = action 

            try:
                # print(next_action)
                full_obs, reward, done, infos = envs2.step(next_action)
            except:
                ipy.embed()

            if knob_noisy:
                obs = add_noise(full_obs, j)
            else:
                obs = full_obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards2.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts2.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts2.obs[-1], rollouts2.recurrent_hidden_states[-1],
                rollouts2.masks[-1]).detach()

        rollouts2.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent2.update(rollouts2)
        rollouts2.after_update()
        value_loss2 = value_loss
        action_loss2 = action_loss
        dist_entropy2 = dist_entropy

        ###################### Logs and storage ########################

        value_loss = (value_loss1 + value_loss2)/2
        action_loss = (action_loss1 + action_loss2)/2
        dist_entropy = (dist_entropy1 + dist_entropy2)/2
        episode_rewards = []
        for ii in range(len(episode_rewards1)):
            episode_rewards.append((episode_rewards1[ii]+episode_rewards2[ii])/2)
        # episode_rewards = episode_rewards1

        writer.add_scalar("Value loss", value_loss, j)
        writer.add_scalar("action loss", action_loss, j)
        writer.add_scalar("dist entropy loss", dist_entropy, j)
        writer.add_scalar("Episode rewards", np.mean(episode_rewards), j)

        if np.mean(episode_rewards) > best_training_reward:
            best_training_reward = np.mean(episode_rewards)
            current_is_best = True
        else:
            current_is_best = False

        # save for every interval-th episode or for the last epoch or for best so far
        if (j % args.save_interval == 0
                or j == num_updates - 1 or current_is_best) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                    actor_critic,
                    None
                ], os.path.join(save_path, args.env_name + "_{}.{}.pt".format(args.save_name,j)))

            if current_is_best:
                torch.save([
                    actor_critic,
                    None
                ], os.path.join(save_path, args.env_name + "_{}.best.pt".format(args.save_name)))
            
            # torch.save([
            #     actor_critic,
            #     getattr(utils.get_vec_normalize(envs1), 'ob_rms', None)
            # ], os.path.join(save_path, args.env_name + "_{}.{}.pt".format(args.save_name,j)))

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
            raise NotImplementedError
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        DR=False # True #Domain Randomization
        ################## for multiprocess world change ######################
        if DR:
            raise NotImplementedError

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

    # if args.algo != 'ipo':
    #     raise NotImplementedError

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


