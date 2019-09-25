from supervised_learning import *
from rl import *

import load_policy
import tf_util

import sys
import getopt
import gym
import pickle
import tensorflow as tf

# Get command line options
optval = getopt.getopt(sys.argv[1:], 'n:e:i:', [])
expert_data_size = 5
num_iter = 10
env_name = ''
for option in optval[0]:
    if option[0] == '-n':
        expert_data_size = int(option[1])
    if option[0] == '-e':
        env_name = option[1]
    if option[0] == '-i':
        num_iter = int(option[1])

# Load expert policy
expert_policy = load_policy.load_policy('experts/' + env_name + ".pkl")
env = gym.make(env_name)

# Define actor policy
sample_action = env.action_space.sample()
sample_obs = env.reset()
policy = NeuralNetwork(len(sample_obs), 20, len(sample_action), 1)

# Dataset aggregation
with tf.Session():
    tf_util.initialize()
    instances = []
    labels = []
    for i in range(num_iter):
        for _ in range(expert_data_size):
            sarss = None
            if i == 0:
                sarss = get_expert_rollout(env, expert_policy, False, max_steps=200)
            else:
                sarss = get_rollout(env, policy, False, max_steps=200)
            instances = instances + [state for state, _, _, _ in sarss]
            labels = labels + [expert_policy(state[None, :])[0] for state, _, _, _ in sarss]
        print("{} labelled pairs after iteration {}".format(len(instances), i))
        learn_nn(instances, labels, policy, lr=0.003)

# Evaluate policy
env = gym.make(env_name)
print('Expected Reward: {}'.format(estimate_policy(env, policy, 20, max_steps=200)))
get_rollout(env, policy, True)
