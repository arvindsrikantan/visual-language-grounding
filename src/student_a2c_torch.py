"""
    Created by arvindsrikantan on 2018-04-25
"""
from a2c_torch import *


class StudentA2C(A2C):
    def __init__(self, model, lr, critic_model, critic_lr, vocab, max_len, teachers, n=20):
        super().__init__(model, lr, critic_model, critic_lr, vocab, max_len, n)
        self.teachers = teachers
        # kl_loss = nn.KLDivLoss(reduce=False)
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.beta = 0.2

        self.custom_student_loss = self.student_loss

    def student_loss(self, target, pred_prob, g, teacher_actor_prob, r_teacher):
        a2c_loss = self.custom_loss(target, pred_prob, g)
        self_entropy = torch.sum(pred_prob * torch.log(pred_prob))
        # kl = 0
        cel = 0
        for i in range(len(teacher_actor_prob)):
            # kl += torch.mean(
            #     r_teacher[i] * torch.sum(kl_loss(torch.log(pred_prob), teacher_actor_prob[i]), dim=1).view(-1, 1))
            # kl += torch.mean(torch.sum(kl_loss(torch.log(pred_prob), teacher_actor_prob[i]), dim=1).view(-1, 1))
            # print("\nA2C Loss: {},  KL Loss : {}".format(a2c_loss.data.numpy().flatten()[0], kl.data.numpy()))
            # _, outputs = torch.max(pred_prob, 1)
            _, targets = torch.max(teacher_actor_prob[i], 1)
            cel += torch.mean(r_teacher[i] * self.ce_loss(pred_prob, targets))
            print("\nA2C Loss: {},  CE Loss : {}".format(a2c_loss.data.numpy().flatten()[0], self.epsilon * torch.mean(r_teacher[i] * self.ce_loss(pred_prob, targets)).data.numpy()))
        return a2c_loss + self.epsilon * ( cel + self.beta * self_entropy )

    def train(self, env, episodes, env_name, gamma=1.0, render=False, reward_scale=1.0, without_mission=False):
        checkpointing = 500
        self.epsilon = 1
        test_rewards = []
        train_rewards = []
        power_gamma = {k: gamma ** k for k in range(10000)}
        for episode in range(episodes + 1):
            if episode % checkpointing == 0:
                # Checkpoint
                self.save_weights(
                    "../pickles/students/%s/checkpoint/%s_n_%s_iter_%s.h5" % (env_name, "%s", self.n, episode))
                test_reward = []
                for _ in range(100):
                    _, _, rewards = self.generate_episode(env, reward_scale)
                    test_reward += [sum(rewards) * reward_scale]
                test_rewards.append((np.array(test_reward).mean(), np.array(test_reward).std()))
                print("Average test rewards = %s" % (str(test_rewards[-1])))
                np.save("../pickles/students/%s/test-rewards/n_%s_iter_%s.npy" % (env_name, self.n, episode),
                        np.array(test_rewards))
            states, actions, rewards = self.generate_episode(env, reward_scale, render=render)
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

            teacher_actions = []
            teacher_rewards = np.zeros((len(self.teachers), len(rewards)))
            for t_i, teacher in enumerate(self.teachers):
                teacher_actions.append(teacher.model(states_transformed).data.numpy())
                t_v = teacher.critic_model(states_transformed).data.numpy().flatten()
                for t in reversed(range(T)):
                    # v_end = 0 if (t + self.n >= T) else t_v[t + self.n]
                    # r_teacher_t = power_gamma[self.n] * v_end + sum(
                    #     [(power_gamma[k] * rewards[t + k] if (t + k < T) else 0) for k in range(self.n)])
                    teacher_rewards[t_i, t] = t_v[t] - v[t]

            self.optimizer.zero_grad()
            self.model.train()
            model_out = self.model(states_transformed)

            loss = self.custom_student_loss(
                Variable(torch.from_numpy(to_categorical(actions, num_classes=env.action_space.n))),
                model_out,
                Variable(torch.from_numpy(g.astype("float32"))),
                Variable(torch.from_numpy(np.array(teacher_actions))),
                Variable(torch.from_numpy(teacher_rewards).float())
            )

            if self.epsilon >=0:
                self.epsilon -= 10**-4
            if self.epsilon < 0:
                self.epsilon = 0
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
            np.save("../pickles/students/%s/n_%s_train-rewards.npy" % (env_name, self.n), np.array(train_rewards))


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
                        default=1, help="The value of gamma in A2C.")

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
        # "../pickles/students/weights",
        "../pickles/students/%s/checkpoint/" % environment_name,
        "../pickles/students/%s/test-rewards/" % environment_name,
        "../pickles/students/%s/test-rewards-lists/" % environment_name
    ]
    createDirectories(dirs)
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = 20  # args.n
    render = args.render
    max_length = 30  # max_len(envs)

    print(
        "Training args: episodes=num_episodes, env_name=%s, render=%s, reward_scale=%s, without_mission=%s, gamma=%s" %
        (environment_name, render, args.reward_scale, args.without_mission, args.gamma)
    )

    vocab = pickle.load(open('../data/vocab.p', 'rb'))

    envs = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-MultiRoom-N2-S4-v0',
            'MiniGrid-Fetch-5x5-N2-v0',
            'MiniGrid-GoToDoor-5x5-v0', 'MiniGrid-PutNear-6x6-N2-v0', 'MiniGrid-LockedRoom-v0']

    # teacher_envs = ['MiniGrid-Fetch-6x6-N2-v0', 'MiniGrid-MultiRoom-N2-S10-v0']
    teacher_envs = ['MiniGrid-MultiRoom-N2-S10-v0']

    teachers = [A2C(
        Actor(environment_name, vocab, max_len=max_length), None,
        Critic(environment_name, vocab, max_len=max_length), None, vocab, max_length
    ) for environment_name in teacher_envs]

    for i, env_name in enumerate(teacher_envs):
        episode = np.load("../pickles/a2c/%s/test-rewards/n_20_iter_100000.npy" % (env_name))[:, 0].argmax() * 500
        load_name = "../pickles/a2c/%s/checkpoint/%s_n_%s_iter_%s.h5" % (env_name, "%s", 20, episode)
        teachers[i].load_weights(load_name)
        teachers[i].model.eval()
        teachers[i].critic_model.eval()

    # Load the actor model from file.
    model = Actor(environment_name, vocab, max_len=max_length)

    # Critic model
    critic_model = Critic(environment_name, vocab, max_len=max_length)

    # critic_model.summary()
    # exit()

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = StudentA2C(model, lr, critic_model, critic_lr, vocab, max_len=max_length, teachers=teachers, n=n)
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

