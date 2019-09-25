from supervised_learning import *
from rl import *

import sys
import getopt
import gym
import pickle

# Get command line options
optval = getopt.getopt(sys.argv[1:], 'e:', [])
env_name = ''
for option in optval[0]:
    if option[0] == '-n':
        expert_data_size = int(option[1])
    if option[0] == '-e':
        env_name = option[1]

# Get expert data
expert_data = pickle.load(open('expert_data/' + env_name + ".pkl", 'rb'))
instances = expert_data['observations']
labels = np.array([action[0] for action in expert_data['actions']])

# Learn imitation policy
policy = NeuralNetwork(instances[0].size, 20, labels[0].size, 1)
learn_nn(instances, labels, policy, lr=0.003)

# Evaluate policy
env = gym.make(env_name)
print('Expected Reward: {}'.format(estimate_policy(env, policy, 20, max_steps=200)))
get_rollout(env, policy, True)
