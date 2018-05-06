"""
The expert recorder.
"""
import argparse
import getch
import random
import gym
import gym_minigrid
import numpy as np
import time
import os

# '''
# 0 Turn left
# 1 Turn right
# 2 Move forward
# 3 Pick up an object
# 4 Drop the object being carried
# 5 Toggle (interact with objects)
# 6 Wait (noop, do nothing)
# '''
#
# BINDINGS = {
#     'a': 0,
#     'd': 1,
#     'w': 2,
#     'p': 3,
#     'q': 4,
#     't': 5,
#     'x': 6
#     }


"""
    left = 0
    right = 1
    forward = 2
    # Toggle/pick up/activate object
    toggle = 3
    # Wait/stay put/do nothing
    wait = 4
"""
BINDINGS = {
    'a': 0,
    'd': 1,
    'w': 2,
    'p': 3,
    'n': 4
}
SHARD_SIZE = 2000


def get_options():
    parser = argparse.ArgumentParser(description='Records an expert..')
    parser.add_argument('--data_directory', type=str, default='data', dest='data_directory',
                        help="The main datastore for this particular expert.")
    parser.add_argument('--env', type=str, dest='env',
                        help="Environment to train this particular expert.")


    args = parser.parse_args()

    return args


def run_recorder(opts):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    ddir = opts.data_directory

    if not os.path.exists(ddir):
        os.makedirs(ddir)

    env_name = opts.env

    # record_history = []  # The state action history buffer.

    env = gym.make(env_name)
    env._max_episode_steps = 1200

    ##############
    # BIND KEYS  #
    ##############

    action = None
    esc = False

    shard_suffix = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    sarsa_pairs = []

    while not esc:

        env.render()
        done = False
        _last_obs = env.reset()
        env.render()
        mission_str = _last_obs['mission']
        print(mission_str)
        while not done:
            env.render()
            # Handle the toggling of different application states

            # Take the current action if a key is pressed.
            action = None
            while action is None:
                keys_pressed = getch.getch()
                if keys_pressed is '+':
                    esc = True
                    break

                pressed = [x for x in BINDINGS if x in keys_pressed]
                action = BINDINGS[pressed[0]] if len(pressed) > 0 else None

            if esc:
                print("ENDING")
                done = True
                break

            obs, reward, done, info = env.step(action)
            env.render()
            mission_str = _last_obs['mission']
            print(mission_str)
            print(reward)
            no_action = False
            sarsa = (_last_obs, action)
            _last_obs = obs
            sarsa_pairs.append(sarsa)

        if esc:
            break

    print("SAVING")
    # Save out recording data.
    num_shards = int(np.ceil(len(sarsa_pairs) / SHARD_SIZE))
    for shard_iter in range(num_shards):
        shard = sarsa_pairs[
                shard_iter * SHARD_SIZE: min(
                    (shard_iter + 1) * SHARD_SIZE, len(sarsa_pairs))]

        shard_name = "{}_{}.npy".format(str(shard_iter), shard_suffix)
        with open(os.path.join(ddir, shard_name), 'wb') as f:
            np.save(f, sarsa_pairs)


if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)
