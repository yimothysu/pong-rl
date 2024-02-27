"""
Inspect the result after preprocessing.
"""

import gym
from matplotlib import pyplot as plt

from preprocess import preprocess

env = gym.make("ALE/Pong-v5", full_action_space=False)

observation, info = env.reset()
res = preprocess(observation)
plt.imshow(res)
plt.show()
print(res.max(), res.min())
