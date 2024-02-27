"""
Vanilla Policy Gradient
"""

import torch
from torchtyping import TensorType


def reward_to_go(rewards: TensorType["H"], discount_factor=0.99) -> TensorType["H"]:
    discount_powers = torch.pow(
        torch.tensor(discount_factor), torch.arange(len(rewards))
    )
    out = torch.flip(torch.cumsum(torch.flip(rewards * discount_powers, [0]), 0), [0])
    return out


def baseline(trajectory_rewards: TensorType["H"]):
    # TODO: Add baseline
    return 0


def compute_advantage(trajectory_rewards: TensorType["H"]) -> torch.Tensor:
    """ """
    return reward_to_go(trajectory_rewards) - baseline(trajectory_rewards)


class Model(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 256)
        self.fc2 = torch.nn.Linear(256, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


class Policy:
    def __init__(self, env, obs_dim, act_dim):
        self.env = env
        self.model = Model(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def _compute_loss(
        self,
        trajectories: tuple[
            TensorType["N", "T", "H", "W", "C"],
            TensorType["N", "T"],
            TensorType["N", "T"],
        ],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)

        n_observations, n_actions, n_rewards = trajectories
        for observations, actions, rewards in zip(n_observations, n_actions, n_rewards):
            advantages = compute_advantage(rewards)
            # print(advantages)
            # print(self.prob(observations.reshape(observations.shape[0], -1), actions))
            loss += (
                torch.log(
                    self.prob(observations.reshape(observations.shape[0], -1), actions)
                )
                * advantages
            ).sum()

        return -loss / n_rewards.shape[0]

    def train(self, trajectories):
        loss = self._compute_loss(trajectories)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def prob(self, observation, action):
        """
        Return probability of taking an action given an observation.
        """
        return self.model(observation)[torch.arange(observation.shape[0]), action]

    def act(self, observation):
        return torch.distributions.Categorical(self.model(observation)).sample().item()
