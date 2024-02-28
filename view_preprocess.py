"""
Inspect the result after preprocessing.
"""

import gym
from matplotlib import pyplot as plt

from preprocess import preprocess

env = gym.make("ALE/Pong-v5", full_action_space=False)

observation, info = env.reset()
prev_res = preprocess(observation)
observation, reward, terminated, truncated, info = env.step(3)
res = preprocess(observation)
plt.imshow(res - prev_res, cmap="gray")
plt.show()
