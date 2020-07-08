# Notes:
# args.procs: number of environments 
# num_frames_per_proc: number of "frames" (i.e., time-steps) that will be run on each environment (default: 128 for PPO)
# num_frames = num_frames_per_proc * num_procs

# exps: experiences in a DictList with a bunch of "attributes", e.g., exps.reward, exps.actions, etc.
# Each attribute has shape (num_frames_per_proc * num_envs, ...). The k-th block of size num_frames_per_proc 
# corresponds to the data collected from the k-th environment. 

# epochs: Number of times collected experiences are used to update parameters.
# batch_size: Size of mini-batch (used to update parameters). The collected experience is broken up into
# mini-batches (default size is 256). By default, there are thus 2048/256 = 8 mini-batches every time
# roll-outs are performed. The number of actual optimization steps performed is thus 8*4 = 32 (4 is the
# default number of epochs).


import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
import math

from gym_minigrid.wrappers import *

import utils
from model import ACModel, ACModel_average

# IPython
import IPython as ipy


def main(raw_args=None):

    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo | ipo (REQUIRED)")
    parser.add_argument("--domain1", required=True,
                        help="name of the first domain to train on (REQUIRED)")
    parser.add_argument("--domain2", required=True,
                        help="name of the second domain to train on (REQUIRED)")
    parser.add_argument("--p1", required=True, type=float,
                        help="Proportion of training environments from first domain (REQUIRED)")
    parser.add_argument("--model", required=True,
                        help="name of the model")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    
    args = parser.parse_args(raw_args)

    args.mem = args.recurrence > 1

    if args.mem:
        raise ValueError("Policies with memory not supported.")

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = args.model

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    torch.backends.cudnn.deterministic = True 
    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments from different domains
    domain1 = args.domain1 # e.g., 'MiniGrid-ColoredKeysRed-v0'
    domain2 = args.domain2 # e.g., 'MiniGrid-ColoredKeysYellow-v0' 

    p1 = args.p1 # Proportion of environments from domain1

    num_envs_total = args.procs # Total number of environments
    num_domain1 = math.ceil(p1*num_envs_total) # Number of environments in domain1
    num_domain2 = num_envs_total - num_domain1 # Number of environments in domain2


    # Environments from domain1
    envs1 = []
    for i in range(num_domain1):
        envs1.append(utils.make_env(domain1, args.seed + 10000 * i))

    # Environments from domain2
    envs2 = []
    for i in range(num_domain2):
        envs2.append(utils.make_env(domain2, args.seed + 10000 * i))

    # All environments
    envs = envs1 + envs2

    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")



    if args.algo == "ipo":
        # Load model for IPO game
        acmodel = ACModel_average(obs_space, envs[0].action_space, args.mem, args.text)
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model loaded\n")
        txt_logger.info("{}\n".format(acmodel))      

    else:    
        # Load model (for standard PPO or A2C)
        acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model loaded\n")
        txt_logger.info("{}\n".format(acmodel))



    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
            txt_logger.info("Optimizer loaded\n")

    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    
        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
            txt_logger.info("Optimizer loaded\n")

    elif args.algo == "ipo":
        # One algo per domain. These have different envivonments, but shared acmodel
        algo1 = torch_ac.IPOAlgo(envs1, acmodel, 1, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)    
        
        algo2 = torch_ac.IPOAlgo(envs2, acmodel, 2, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)   

        if "optimizer_state1" in status:
            algo1.optimizer.load_state_dict(status["optimizer_state1"])
            txt_logger.info("Optimizer 1 loaded\n")
        if "optimizer_state2" in status:
            algo2.optimizer.load_state_dict(status["optimizer_state2"])
            txt_logger.info("Optimizer 2 loaded\n")

    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))


    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames: 
        # Update model parameters

        update_start_time = time.time()

        if args.algo == "ipo":

            # Standard method

            # Collect experiences on first domain
            exps1, logs_exps1 = algo1.collect_experiences()

            # Update params of model corresponding to first domain
            logs_algo1 = algo1.update_parameters(exps1)

            # Collect experiences on second domain
            exps2, logs_exps2 = algo2.collect_experiences()

            # Update params of model corresponding to second domain
            logs_algo2 = algo2.update_parameters(exps2)

            # Update end time 
            update_end_time = time.time()

            # Combine logs
            logs_exps = {'return_per_episode': logs_exps1["return_per_episode"] + logs_exps2["return_per_episode"],
                        'reshaped_return_per_episode': logs_exps1["reshaped_return_per_episode"] + logs_exps2["reshaped_return_per_episode"],
                        'num_frames_per_episode': logs_exps1["num_frames_per_episode"] + logs_exps2["num_frames_per_episode"],
                        'num_frames': logs_exps1["num_frames"] + logs_exps2["num_frames"]}
            
            logs_algo = {'entropy': (num_domain1*logs_algo1["entropy"] + num_domain2*logs_algo2["entropy"])/num_envs_total,
                        'value': (num_domain1*logs_algo1["value"] + num_domain2*logs_algo2["value"])/num_envs_total,
                        'policy_loss': (num_domain1*logs_algo1["policy_loss"] + num_domain2*logs_algo2["policy_loss"])/num_envs_total,
                        'value_loss': (num_domain1*logs_algo1["value_loss"] + num_domain2*logs_algo2["value_loss"])/num_envs_total,
                        'grad_norm': (num_domain1*logs_algo1["grad_norm"] + num_domain2*logs_algo2["grad_norm"])/num_envs_total}

            logs = {**logs_exps, **logs_algo}
            num_frames += logs["num_frames"]

        else:    
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()
            num_frames += logs["num_frames"]
        
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            # header += ["debug_last_env_reward"]
            # data += [logs["debug_last_env_reward"]]

            header += ["total_loss"]
            data += [logs["policy_loss"] - args.entropy_coef*logs["entropy"] + args.value_loss_coef*logs["value_loss"]]

            if status["num_frames"] == 0:
                csv_logger.writerow(header)

            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:

            if args.algo == "ipo":
                status = {"num_frames": num_frames, "update": update,
                          "model_state": acmodel.state_dict(), "optimizer_state1": algo1.optimizer.state_dict(),
                          "optimizer_state2": algo2.optimizer.state_dict()}
            else:    
                status = {"num_frames": num_frames, "update": update,
                          "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")


# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()            
