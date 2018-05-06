"""
    Created by arvindsrikantan on 2018-04-13
"""
import sys

import numpy as np

from a2c_torch import main
# from student_a2c_torch import main

if __name__ == '__main__':
    env_name = "MiniGrid-Fetch-6x6-N2-v0"
    episode = np.load("../pickles/a2c/%s/test-rewards/n_20_iter_100000.npy" % (env_name))[:, 0].argmax() * 500
    print("Loading episode: %s, reward: %s" % (episode, np.load("../pickles/a2c/%s/test-rewards/n_20_iter_100000.npy" % (env_name))[:, 0].max()))
    # load_models = "../pickles/a2c/%s/checkpoint/%s_n_%s_iter_%s.h5" % (env_name, "%s", 20, episode)
    load_models = None
    print("Using pretrained %s" % load_models)
    main(sys.argv, load_models)
