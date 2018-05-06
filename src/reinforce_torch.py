"""
    Created by arvindsrikantan on 2018-04-23
"""
import argparse
import os
import sys

import gym
# import keras
# import keras.backend as K
import matplotlib
import numpy as np
import torch
import time

from torch import optim

matplotlib.use('Agg')
# seaborn.set()
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, vocab, max_len):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.vocab = vocab
        self.max_len = max_len

        def loss_func(target, pred, g):
            loss = torch.mean(g * - torch.log(torch.sum(target * pred, dim=1)))
            return loss

        self.custom_loss = loss_func
        # self.model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=['accuracy'])

    def pad_seq(self, instr):
        padded = instr
        if len(instr) < self.max_len:
            padded = list(np.zeros(self.max_len - len(instr)))
            # instr.extend(padding)
            padded.extend(instr)

        return padded

    def get_state(self, state, batch=False, without_mission=False):
        mission = [self.vocab[word] for word in state['mission'].split()]
        mission = self.pad_seq(mission)
        # mission = sequence.pad_sequences([mission], maxlen=12)
        if batch:
            if without_mission:
                return state['image']
            return [state['image'].astype("float32"), np.array([mission]).astype("long")]
        if without_mission:
            return np.array([state['image'].astype("float32")])
        return [np.array([state['image'].astype("float32")]), np.array([mission]).astype("long")]

    def train(self, env, episodes, env_name, gamma=1.0, render=False, reward_scale=1.0):
        # Trains the model on a single episode using REINFORCE.
        checkpointing = 500
        # episodes = 60000
        test_rewards = []
        train_rewards = []
        power_gamma = {k: gamma ** k for k in range(10000)}
        for episode in range(episodes + 1):
            if episode % checkpointing == 0:
                # Checkpoint
                self.save_weights("pickles/reinforce/checkpoint/iter_%s.h5" % episode)
                test_reward = 0
                for _ in range(100):
                    _, _, rewards = self.generate_episode(env, reward_scale)
                    test_reward += sum(rewards) * reward_scale
                test_rewards.append(test_reward / reward_scale)
                print("Average test rewards = %s" % (test_rewards[-1]))
                np.save("pickles/reinforce/test-rewards/iter_%s.npy" % episode, np.array(test_rewards))
            states, actions, rewards = self.generate_episode(env, reward_scale, render=render)

            T = len(rewards)
            g = np.zeros(T)
            for t in reversed(range(T)):
                g[t] = sum([power_gamma[k] * rewards[k] for k in range(t, T)])

            y = np.array(np_utils.to_categorical(actions, num_classes=env.action_space.n))
            history = self.model.fit(np.array(states), y, epochs=1, batch_size=len(states), verbose=False,
                                     sample_weight=g)

            print("Episode %6d's, Steps = %3d, loss = %+.5f, cumulative reward:%+5.5f" % (
            episode, len(states), history.history['loss'][0], sum(rewards) * reward_scale))
            train_rewards.append(sum(rewards) * reward_scale)
            np.save("pickles/reinforce/train-rewards.npy", np.array(train_rewards))

    def generate_episode(self, env, reward_scale, render=False, without_mission=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        self.model.eval()

        terminal = False
        state = env.reset()
        action_space = np.array(range(env.action_space.n))
        while not terminal:
            states.append(self.get_state(state, batch=True, without_mission=without_mission))
            if render:
                env.render()
                time.sleep(0.1)
            action = np.random.choice(action_space, 1, p=self.model(self.get_state(state, without_mission=without_mission)).data.numpy().flatten())[0]
            state, reward, terminal, _ = env.step(action)

            actions.append(action)
            rewards.append(reward/reward_scale)

        return states, actions, rewards

    def save_weights(self, name):
        torch.save(self.model.state_dict(), name)

    def load_weights(self, name):
        checkpoint = torch.load(name)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=100000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def createDirectories(l):
    for l in l:
        if not os.path.exists(l):
            os.makedirs(l)


def get_test_rewards(env, model):
    checkpointing = 500

    for episode in range(0, 100000, checkpointing):
        print("Episode: %s" % episode)
        model.load_weights("pickles/reinforce/checkpoint/iter_%s.h5" % (episode))
        test_rewards = []
        for _ in range(100):
            _, _, rewards = model.generate_episode(env)
            test_rewards.append(np.array(rewards))
        np.save("pickles/reinforce/test-rewards/iter_%s.npy" % episode, np.array(test_rewards))


def plot():
    r = []
    y, err = list(zip(*[r[i] for i in range(len(r)) if i % 1 == 0]))
    x = list(range(0, len(y) * 500, 500))
    # x = list(range(0, len(y)*500, 500))
    plt.figure()
    plt.errorbar(x, y, yerr=err)
    plt.xlabel("Training episodes")
    plt.ylabel("Average reward over 100 episodes")
    plt.title("REINFORCE cumulative rewards averaged over 100 test episodes")
    plt.savefig("reinforce_500_no_seaborn.png")


def test_reward_with_error():
    checkpointing = 500
    test_rewards_mean = []
    test_rewards_sd = []

    for episode in range(0, 60000, checkpointing):
        # print("Episode: %s" % episode)
        test_rewards = np.array(
            [_temp.sum() * 100 for _temp in np.load("pickles/reinforce/test-rewards/iter_%s.npy" % (episode))])
        test_rewards_mean.append(test_rewards.mean())
        test_rewards_sd.append(test_rewards.std())

    return test_rewards_mean, test_rewards_sd


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    dirs = [
        # "pickles/weights",
        "pickles/reinforce/checkpoint/",
        "pickles/reinforce/test-rewards-lists/",
        "pickles/reinforce/test-rewards/"
    ]
    createDirectories(dirs)
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    rl_model = Reinforce(model, lr)
    # rl_model.train(env, num_episodes, gamma=1, render=False)
    # rl_model.save_weights("./pickles/weights/reinforce.h5")
    # rl_model.load_weights("./pickles/weights/reinforce.h5")
    # _, _, rewards = rl_model.generate_episode(env, True)
    # print("Test reward = %s" % (sum(rewards) * 100))

    # get_test_rewards(env, rl_model)

    # print(list(zip(*test_reward_with_error())))
    plot()


if __name__ == '__main__':
    main(sys.argv)
