import gym
import numpy as np

from policy import Policy
from preprocess import preprocess


env = gym.make("ALE/Pong-v5", full_action_space=False)

obs_dim = preprocess(np.zeros((env.observation_space.shape))).numel()
act_dim = env.action_space.n

policy = Policy(env, obs_dim, act_dim)
policy.load("model.pt")


env.close()

env = gym.make("ALE/Pong-v5", full_action_space=False, render_mode="human")
for _ in range(10):
    observation, info = env.reset()
    prev_observation = preprocess(observation)
    for _ in range(100):
        env.render()
        observation = preprocess(observation)
        action = policy.act((observation - prev_observation).flatten())
        prev_observation = observation
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

env.close()
