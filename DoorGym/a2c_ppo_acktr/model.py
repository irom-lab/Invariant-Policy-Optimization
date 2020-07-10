import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

from a2c_ppo_acktr.utils import init, AddBias

# from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian # , DiagGaussian_av
from a2c_ppo_acktr.distributions import * # , DiagGaussian_av


import IPython as ipy

logits_input = False
knob_pos_smoothening = False


# #
# # Standardize distribution interfaces
# #

# # Normal
# FixedNormal = torch.distributions.Normal

# log_prob_normal = FixedNormal.log_prob
# FixedNormal.log_probs = lambda self, actions: log_prob_normal(
#     self, actions).sum(
#         -1, keepdim=True)

# normal_entropy = FixedNormal.entropy
# FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

# FixedNormal.mode = lambda self: self.mean


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.tt = 0
        self.nn = 0
        self.visionmodel = None
        self.knob_target_hist = torch.zeros(1,3).cuda()
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def obs2img_vec(self, inputs):
        img_size = 256
        joints_nn = self.nn
        joints = inputs[:,:joints_nn*2]
        finger_tip_target = inputs[:,joints_nn*2:joints_nn*2+3]
        img_front = inputs[:,joints_nn*2+3:-3*img_size*img_size].view(-1, 3, img_size, img_size)
        img_top   = inputs[:,-3*img_size*img_size:].view(-1, 3, img_size, img_size)
        return joints, finger_tip_target, img_front, img_top

    def obs2inputs(self, inputs, epoch):
        joints, finger_tip_target, img_front, img_top = self.obs2img_vec(inputs)

        with torch.no_grad():
            pp, hm1, hm2 = self.visionmodel(img_top, img_front)        
        knob_target = pp
        dist_vec = finger_tip_target - knob_target

        inputs = torch.cat((joints, dist_vec), 1)
        return inputs


# Average policy for IPO
class Policy_av(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy_av, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                raise NotImplementedError
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase_av
            else:
                raise NotImplementedError

        self.tt = 0
        self.nn = 0
        self.visionmodel = None
        self.knob_target_hist = torch.zeros(1,3).cuda()
        self.base = base(obs_shape[0], action_space.shape[0], **base_kwargs)

        # if action_space.__class__.__name__ == "Discrete":
        #     raise NotImplementedError
        #     num_outputs = action_space.n
        #     self.dist = Categorical(self.base.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "Box":
        #     num_outputs = action_space.shape[0]
        #     self.dist = DiagGaussian_av(self.base.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "MultiBinary":
        #     raise NotImplementedError
        #     num_outputs = action_space.shape[0]
        #     self.dist = Bernoulli(self.base.output_size, num_outputs)
        # else:
        #     raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, dist, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, dist, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def obs2img_vec(self, inputs):
        img_size = 256
        joints_nn = self.nn
        joints = inputs[:,:joints_nn*2]
        finger_tip_target = inputs[:,joints_nn*2:joints_nn*2+3]
        img_front = inputs[:,joints_nn*2+3:-3*img_size*img_size].view(-1, 3, img_size, img_size)
        img_top   = inputs[:,-3*img_size*img_size:].view(-1, 3, img_size, img_size)
        return joints, finger_tip_target, img_front, img_top

    def obs2inputs(self, inputs, epoch):
        joints, finger_tip_target, img_front, img_top = self.obs2img_vec(inputs)

        with torch.no_grad():
            pp, hm1, hm2 = self.visionmodel(img_top, img_front)        
        knob_target = pp
        dist_vec = finger_tip_target - knob_target

        inputs = torch.cat((joints, dist_vec), 1)
        return inputs





class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


# Create base for average policy for IPO
class MLPBase_av(NNBase):
    def __init__(self, num_inputs, num_actions, recurrent=False, hidden_size=64):
        super(MLPBase_av, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            raise NotImplementedError
            num_inputs = hidden_size


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.actor1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.actor2 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic2 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear1 = init_(nn.Linear(hidden_size, 1))

        self.critic_linear2 = init_(nn.Linear(hidden_size, 1))


        # Action distribution
        init_dist_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean1 = init_dist_(nn.Linear(hidden_size, num_actions))
        self.logstd1 = AddBias(torch.zeros(num_actions))

        self.fc_mean2 = init_dist_(nn.Linear(hidden_size, num_actions))
        self.logstd2 = AddBias(torch.zeros(num_actions))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        # Critic
        hidden_critic1 = self.critic1(x)
        value1 = self.critic_linear1(hidden_critic1)

        hidden_critic2 = self.critic2(x)
        value2 = self.critic_linear2(hidden_critic2)

        critic_output = (value1+value2)/2

        # Actor
        hidden_actor1 = self.actor1(x)
        action_mean1 = self.fc_mean1(hidden_actor1)

        hidden_actor2 = self.actor2(x)
        action_mean2 = self.fc_mean2(hidden_actor2)

        #  An ugly hack for KFAC implementation.
        zeros = torch.zeros(action_mean1.size())
        if hidden_actor1.is_cuda:
            device = hidden_actor1.get_device()
            zeros = zeros.to(device)

        action_logstd1 = self.logstd1(zeros)
        action_logstd2 = self.logstd2(zeros)

        action_mean_av = (action_mean1+action_mean2)/2
        # action_logstd_av = (action_logstd1+action_logstd2)/2

        action_std_av = (action_logstd1.exp() + action_logstd2.exp())/2

        actor_dist = FixedNormal(action_mean_av, action_std_av)

        return critic_output, actor_dist, rnn_hxs
