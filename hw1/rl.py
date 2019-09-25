import random
import numpy as np


# Compute a single rollout.
#
# env: Environment
# policy: Policy
# render: bool
# return: [(np.array([state_dim]), np.array([action_dim]), float, np.array([state_dim]))]
#          ((state, action, reward, next_state) tuples)
def get_rollout(env, policy, render, max_steps=10000):
    # Step 1: Initialization
    state = env.reset()
    done = False
    steps = 0

    # Step 2: Compute rollout
    sarss = []
    while not done:
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy(state).detach().numpy()

        # Step 2c: Transition environment
        next_state, reward, done, _ = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sarss.append((state, action, reward, next_state))

        # Step 2e: Update state
        state = next_state

        # Termination condition
        steps += 1
        if steps >= max_steps:
            break

    # Step 3: Render final state
    if render:
        env.render()

    return sarss


# Estimate the cumulative reward of the policy.
#
# env: Environment
# policy: Policy
# n_rollouts: int
# return: float
def estimate_policy(env, policy, n_rollouts, max_steps=10000):
    cum_reward = 0.0
    for i in range(n_rollouts):
        sarss = get_rollout(env, policy, False, max_steps)
        tmp_rew = np.sum(np.array([r for _, _, r, _ in sarss]))
        cum_reward += tmp_rew
    return cum_reward / n_rollouts


def get_expert_rollout(env, policy, render, max_steps=10000):
    # Step 1: Initialization
    state = env.reset()
    done = False
    steps = 0

    # Step 2: Compute rollout
    sarss = []
    while not done:
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy(state[None, :])[0]

        # Step 2c: Transition environment
        next_state, reward, done, _ = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sarss.append((state, action, reward, next_state))

        # Step 2e: Update state
        state = next_state

        # Termination condition
        steps += 1
        if steps >= max_steps:
            break

    # Step 3: Render final state
    if render:
        env.render()

    return sarss
