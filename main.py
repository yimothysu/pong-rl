import gym

env = gym.make("ALE/Pong-v5", full_action_space=False, render_mode="human")
observation, info = env.reset(seed=42)
