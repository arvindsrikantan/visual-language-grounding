from a2c_torch import Actor
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import argparse
import numpy as np
import pickle
import os
import pdb
import sys
import gym


max_len = 30
vocab = vocab = pickle.load(open('../data/vocab.p', 'rb'))

def pad_seq(instr):
    padded = instr
    if len(instr) < max_len:
        padded = list(np.zeros(max_len - len(instr)))
        # instr.extend(padding)
        padded.extend(instr)

    return padded


def get_state(state, batch=False, without_mission=False):
    mission = [vocab[word] for word in state['mission'].split()]
    mission = pad_seq(mission)
    # mission = sequence.pad_sequences([mission], maxlen=12)
    if batch:
        if without_mission:
            return state['image']
        return [state['image'].astype("float32"), np.array([mission]).astype("long")]
    if without_mission:
        return np.array([state['image'].astype("float32")])
    return [np.array([state['image'].astype("float32")]), np.array([mission]).astype("long")]


class Imitation():
    def __init__(self, environment_name, vocab, lr):
        # Load the expert model.
        max_length = 30  # max_len(envs)

        # Load the actor model from file.
        self.model = Actor(environment_name, vocab, max_len=max_length)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def run_expert(self, env, render=False):
        # Generates an episode by running the expert policy on the given env.
        return Imitation.generate_episode(self.expert, env, render)

    def run_model(self, env, render=False):
        # Generates an episode by running the cloned policy on the given env.
        return Imitation.generate_episode(self.model, env, render)

    def _save_model_weights(self, model_weights_path):
        torch.save(self.model.state_dict(), model_weights_path)

    def load_model_weights(self, model_weights_path):
        checkpoint = torch.load(model_weights_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])

    @staticmethod
    def generate_episode(model, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        state = env.reset()
        terminal = False
        im_state, txt_state = get_state(state)

        while not terminal:
            # render if set to True
            if render:
                env.render()
            action_softmax = model([im_state, txt_state])
            action = action_softmax.data.numpy().argmax()
            next_state, reward, terminal, d = env.step(action)
            # pdb.set_trace()

            # # add to dataset
            # states.append(cur_state)
            # actions.append(action)
            # rewards.append(reward)

            # update cur_state for next iter
            im_state, txt_state = get_state(next_state)

        return states, actions, rewards

    def run_episodes(self, env, num_episodes=1, render=False):
        # Runs model in an env for num_episodes

        data = {'states': [], 'actions': [], 'rewards': []}

        for episode in range(num_episodes):
            states, actions, rewards = self.run_expert(env, render)
            data['states'].extend(states)
            data['actions'].extend(actions)
            data['rewards'].extend(rewards)

        pickle.dump(data, open('data/data_{}_episodes.p'.format(num_episodes), 'wb'))

    def train(self, env, epochs=10):#, env, X, y, num_episodes=100, num_epochs=50, render=False):
        # Trains the model on training data generated by the expert policy.
        # Args:
        # - env: The environment to run the expert policy on.
        # - num_episodes: # episodes to be generated by the expert.
        # - num_epochs: # epochs to train on the data generated by the expert.
        # - render: Whether to render the environment.
        # Returns the final loss and accuracy.
        # TODO: Implement this method. It may be helpful to call the class
        #       method run_expert() to generate training data.

        # load the data
        data_dir = 'gotodoor'
        states_im = []
        states_txt = []
        actions = []


        for ifile in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir,ifile))

            for state, action in data:
                im, txt = get_state(state, batch=True)
                states_im.append(im)
                states_txt.append(txt)
                actions.append(action)

        states_im = np.array(states_im)
        states_txt = np.array(states_txt)
        actions = np.array(actions)
        indices = np.arange(states_im.shape[0])

        for epoch in range(epochs):
            idx = np.random.choice(indices, len(indices))
            states_im = states_im[idx]
            states_txt = states_txt[idx]
            actions = actions[idx]

            self.optimizer.zero_grad()
            self.model.train()
            # do the forward
            model_out = self.model([states_im,states_txt])
            targets = Variable(torch.from_numpy(actions))#to_categorical(actions, num_classes=env.action_space.n)))

            loss = self.loss(
                model_out, targets
                )

            loss.backward()
            self.optimizer.step()
            print("Epoch : {} , Loss : {}".format(epoch,loss.data[0]))

        self._save_model_weights('expert_gtd.p')


    def run_policy(self, env, num_episodes=50, render=False):
        # Runs policy of model_type in env for num_episodes
        # and returns mean and std over num_episodes
        tot_rewards = []
        for episode in range(num_episodes):
            _, actions, rewards = self.run_model(env, render)

            tot_rewards.append(sum(rewards))

        tot_rewards = np.array(tot_rewards)
        # print('Total rewards : ', tot_rewards)
        return np.mean(tot_rewards), np.std(tot_rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--expert-pickles-path', dest='expert_weights_path',
                        type=str, default='LunarLander-v2-weights.h5',
                        help="Path to the expert pickles file.")

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


def main():
    # Parse command-line arguments.
    # args = parse_arguments()
    # model_config_path = args.model_config_path
    # expert_weights_path = args.expert_weights_path
    # render = args.render

    # Create the environment.
    env_name = 'MiniGrid-GoToDoor-5x5-v0'
    env = gym.make(env_name)
    vocab = pickle.load(open('../data/vocab.p', 'rb'))
    lr = 1e-3
    im_agent = Imitation(env_name,vocab,lr)
    im_agent.train(env, epochs=100)
    # im_agent.run_policy(env, render=True)



if __name__ == '__main__':
    main()
