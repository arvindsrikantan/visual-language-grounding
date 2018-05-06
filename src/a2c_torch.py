"""
    Created by arvindsrikantan on 2018-04-23
"""
import os

import gym_minigrid

gym_minigrid

from reinforce_torch import Reinforce

import argparse
import sys
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np

import pickle


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float32')[y]


class Actor(nn.Module):
    def __init__(self, env_name, vocab, max_len, with_language=True):
        super(Actor, self).__init__()
        self.temp_env = gym.make(env_name)
        self.inp_shape = self.temp_env.observation_space.spaces['image'].shape
        self.with_language = with_language
        if self.inp_shape is None:
            self.inp_shape = self.temp_env.observation_space.spaces['image'].spaces['image'].shape

        self.conv = nn.Conv2d(self.inp_shape[-1], 4, (2, 2))
        self.linear = nn.Linear(36, 24)

        if self.with_language:
            self.embedding = nn.Embedding(len(vocab), 24)
            self.lstm = nn.LSTM(24, 12, batch_first=True)
            self.linear = nn.Linear(36 + (max_len * 12), 24)

        self.dense = nn.Linear(24, self.temp_env.action_space.n)

    def forward(self, inputs):
        # might run into problems cos of np resize
        vis_input, txt_input = Variable(torch.from_numpy(inputs[0].transpose((0, 3, 1, 2)))), \
                               Variable(torch.from_numpy(np.array(inputs[1]).reshape((np.array(inputs[1]).shape[0],-1))))
        cnn_outputs = F.max_pool2d(F.relu(self.conv(vis_input)), (2, 2))
        flat_cnn_out = cnn_outputs.view(cnn_outputs.shape[0], -1)
        linear_in = flat_cnn_out

        if self.with_language:
            embed_out = self.embedding(txt_input)
            lstm_out, _ = self.lstm(embed_out)
            concat_out = torch.cat((flat_cnn_out, lstm_out.contiguous().view(txt_input.shape[0], -1)), dim=1)
            linear_in = concat_out

        linear1_out = F.relu(self.linear(linear_in))
        out = F.softmax(self.dense(linear1_out))

        return out


class Critic(nn.Module):
    def __init__(self, env_name, vocab, max_len, with_language=True):
        super(Critic, self).__init__()
        self.temp_env = gym.make(env_name)
        self.with_language = with_language
        self.inp_shape = self.temp_env.observation_space.spaces['image'].shape
        if self.inp_shape is None:
            self.inp_shape = self.temp_env.observation_space.spaces['image'].spaces['image'].shape

        self.conv = nn.Conv2d(self.inp_shape[-1], 4, (2, 2))
        self.linear1 = nn.Linear(36, 24)

        if self.with_language:
            self.embedding = nn.Embedding(len(vocab), 24)
            self.lstm = nn.LSTM(24, 12, batch_first=True)
            self.linear1 = nn.Linear(36 + (max_len * 12), 24)

        self.linear2 = nn.Linear(24, 30)

        self.dense = nn.Linear(30, 1)

    def forward(self, inputs):
        # might run into problems cos of np resize
        vis_input, txt_input = Variable(torch.from_numpy(inputs[0].transpose((0, 3, 1, 2)))), \
                               Variable(torch.from_numpy(inputs[1].reshape((inputs[1].shape[0],-1))))
        cnn_outputs = F.max_pool2d(F.relu(self.conv(vis_input)), (2, 2))
        flat_cnn_out = cnn_outputs.view(cnn_outputs.shape[0], -1)
        linear_in = flat_cnn_out

        if self.with_language:
            embed_out = self.embedding(txt_input)
            lstm_out, _ = self.lstm(embed_out)
            concat_out = torch.cat((flat_cnn_out, lstm_out.contiguous().view(txt_input.shape[0], -1)), dim=1)
            linear_in = concat_out

        linear1_out = F.relu(self.linear1(linear_in))
        linear2_out = F.relu(self.linear2(linear1_out))
        out = self.dense(linear2_out)

        return out


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, vocab, max_len, n=20):
        """
        Initializes A2C.
        :param model: The actor model.
        :param lr:  Learning rate for the actor model.
        :param critic_model: The critic model.
        :param critic_lr: Learning rate for the critic model.
        :param n: The value of N in N-step A2C.
        """
        self.critic_model = critic_model
        self.max_len = max_len
        self.n = n
        self.lr = lr
        super().__init__(model, lr, vocab, max_len)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_lr)

        def mse_loss_func(pred, target):
            loss = torch.sum((pred - target) ** 2)
            return loss

        self.critic_loss = nn.MSELoss(size_average=True)  # mse_loss_func

    def train(self, env, episodes, env_name, gamma=1.0, render=False, reward_scale=1.0, without_mission=False):
        checkpointing = 500
        test_rewards = []
        train_rewards = []
        power_gamma = {k: gamma ** k for k in range(10000)}
        for episode in range(episodes + 1):
            if episode % checkpointing == 0:
                # Checkpoint
                self.save_weights("../pickles/a2c/%s/checkpoint/%s_n_%s_iter_%s.h5" % (env_name, "%s", self.n, episode))
                test_reward = []
                for _ in range(100):
                    _, _, rewards = self.generate_episode(env, reward_scale)
                    test_reward += [sum(rewards) * reward_scale]
                test_rewards.append((np.array(test_reward).mean(), np.array(test_reward).std()))
                print("Average test rewards = %s" % (str(test_rewards[-1])))
                np.save("../pickles/a2c/%s/test-rewards/n_%s_iter_%s.npy" % (env_name, self.n, episode),
                        np.array(test_rewards))
            states, actions, rewards = self.generate_episode(env, reward_scale, render=render)
            if sum(rewards)>0:
                val = 0
            r = np.zeros(len(rewards))
            g = np.zeros(len(rewards))
            T = len(rewards)
            if without_mission:
                states_transformed = np.array(states)
            else:
                im, descr = zip(*states)
                # descr = self.padding(descr)
                states_transformed = [np.array(im), np.array(descr)]
            self.critic_model.eval()
            v = self.critic_model(states_transformed).data.numpy().flatten()
            for t in reversed(range(T)):
                v_end = 0 if (t + self.n >= T) else v[t + self.n]
                r[t] = power_gamma[self.n] * v_end + sum(
                    [(power_gamma[k] * rewards[t + k] if (t + k < T) else 0) for k in range(self.n)])
                g[t] = r[t] - v[t]

            self.optimizer.zero_grad()
            self.model.train()
            model_out = self.model(states_transformed)
            loss = self.custom_loss(
                Variable(torch.from_numpy(to_categorical(actions, num_classes=env.action_space.n))),
                model_out,
                Variable(torch.from_numpy(g.astype("float32")))
            )
            loss.backward()
            self.optimizer.step()

            self.critic_optimizer.zero_grad()
            self.critic_model.train()
            critic_model_out = self.critic_model(states_transformed)
            critic_loss = self.critic_loss(critic_model_out, Variable(torch.from_numpy(r.astype("float32"))))
            critic_loss.backward()
            self.critic_optimizer.step()

            print("Episode %6d's, Steps = %3d, loss = %+.5f, critic_loss = %+.5f, cumulative reward:%+5.5f" % (
                episode, len(states), loss.data[0], critic_loss.data[0],
                sum(rewards) * reward_scale))
            train_rewards.append(sum(rewards) * reward_scale)
            np.save("../pickles/a2c/%s/n_%s_train-rewards.npy" % (env_name, self.n), np.array(train_rewards))

    def padding(self, instructions):
        out = []
        for instr in instructions:
            if len(instr) < self.max_len:
                padding = list(np.zeros(self.max_len - len(instr)))
                instr.extend(padding)
            out.append(instr)

        return out

    def save_weights(self, name):
        torch.save(self.model.state_dict(), name % "actor")
        torch.save(self.critic_model.state_dict(), name % "critic")

    def load_weights(self, name):
        checkpoint = torch.load(name % "actor")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        checkpoint = torch.load(name % "critic")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.critic_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.critic_model.load_state_dict(checkpoint)


def createDirectories(l):
    for l in l:
        if not os.path.exists(l):
            os.makedirs(l)


def max_len(envs):
    max_length = 0
    for env_name in envs:
        env = gym.make(env_name)
        for i in range(100000):
            state = env.reset()
            max_length = max(max_length, len(state['mission']))
    return max_length


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment-name', dest='environment_name',
                        type=str, default='MiniGrid-Fetch-6x6-N2-v0',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=100000, help="Number of episodes to train on.")
    parser.add_argument('--reward-scale', dest='reward_scale', type=float,
                        default=1, help="The scale factor for rewards")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")  # 5e-4 before
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="The value of gamma in A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--with-mission', dest='without_mission',
                              action='store_false',
                              help="Whether to use the mission string.")
    parser_group.add_argument('--without-mission', dest='without_mission',
                              action='store_true',
                              help="Whether to use the mission string.")
    parser.set_defaults(without_mission=False)

    return parser.parse_args()


def main(args, load_models=None):
    # Parse command-line arguments.
    args = parse_arguments()

    environment_name = args.environment_name
    print("Running env: %s, with reward scaling of: %s" % (environment_name, args.reward_scale))
    # Create the environment.
    env = gym.make(environment_name)
    dirs = [
        # "../pickles/a2c/weights",
        "../pickles/a2c/%s/checkpoint/" % environment_name,
        "../pickles/a2c/%s/test-rewards/" % environment_name,
        "../pickles/a2c/%s/test-rewards-lists/" % environment_name
    ]
    createDirectories(dirs)
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = 20  # args.n
    render = args.render

    print(
        "Training args: episodes=num_episodes, env_name=%s, render=%s, reward_scale=%s, without_mission=%s, gamma=%s" %
        (environment_name, render, args.reward_scale, args.without_mission, args.gamma))

    vocab = pickle.load(open('../data/vocab.p', 'rb'))

    envs = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-MultiRoom-N2-S4-v0',
            'MiniGrid-Fetch-5x5-N2-v0',
            'MiniGrid-GoToDoor-5x5-v0', 'MiniGrid-PutNear-6x6-N2-v0', 'MiniGrid-LockedRoom-v0']
    max_length = 30  # max_len(envs)

    # Load the actor model from file.
    model = Actor(environment_name, vocab, max_len=max_length)
    if torch.cuda.is_available():
        model = model.cuda()

    # Critic model
    critic_model = Critic(environment_name, vocab, max_len=max_length)
    if torch.cuda.is_available():
        critic_model = critic_model.cuda()

    # critic_model.summary()
    # exit()

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = A2C(model, lr, critic_model, critic_lr, vocab, max_len=max_length, n=n)
    if load_models is not None:
        a2c.load_weights(load_models)
        print("Loaded")

    a2c.train(env, episodes=num_episodes, env_name=environment_name, render=render, reward_scale=args.reward_scale,
              without_mission=args.without_mission, gamma=args.gamma)

    # for _n in [1, 20, 50, 100]:
    #     print("Starting for n=%s" % _n)
    #     get_test_rewards(env, a2c, n=_n)
    #
    # for _n in [1, 20, 50, 100]:
    #     print("Starting for n=%s" % _n)
    #     print(list(zip(*test_reward_with_error(n=_n))))

    # plot()


if __name__ == '__main__':
    main(sys.argv)
