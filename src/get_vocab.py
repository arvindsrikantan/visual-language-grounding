import pickle

import gym
import gym_minigrid

gym_minigrid
envs = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-MultiRoom-N2-S4-v0', 'MiniGrid-Fetch-5x5-N2-v0',
        'MiniGrid-GoToDoor-5x5-v0', 'MiniGrid-PutNear-6x6-N2-v0', 'MiniGrid-LockedRoom-v0']
s = set()
d = {}
for env_name in envs:
    env = gym.make(env_name)
    for i in range(100000):
        state = env.reset()
        for word in state['mission'].split():
            s.add(word)
words = list(s)

for i in range(len(words)):
    d[words[i]] = i

pickle.dump(d, open('vocab.p', 'wb'))
