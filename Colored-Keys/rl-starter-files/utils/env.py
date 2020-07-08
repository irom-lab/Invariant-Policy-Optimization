import gym
import gym_minigrid
import gym_generalization

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env
