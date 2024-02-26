"""
Vanilla Policy Gradient
"""

import torch
from torchtyping import TensorType


def reward_to_go(rewards: TensorType["H"], discount_factor=0.99) -> TensorType["H"]:
    return torch.tensor(
        [
            torch.sum(
                torch.tensor(
                    [
                        discount_factor ** (i - t) * rewards[i]
                        for i in range(t, len(rewards))
                    ]
                )
            )
            for t in range(len(rewards))
        ]
    )


def baseline(trajectory_rewards: TensorType["H"]):
    # TODO: Add baseline
    return 0


def advantage(trajectory_rewards: TensorType["H"]) -> torch.Tensor:
    """ """
    return reward_to_go(trajectory_rewards) - baseline(trajectory_rewards)


class Model(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 128)
        self.fc2 = torch.nn.Linear(128, act_dim)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


class Policy:
    def __init__(self, env, obs_dim, act_dim):
        self.env = env
        self.model = Model(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def _compute_loss(self, trajectories: TensorType["N", "3", "H"]) -> torch.Tensor:
        return -torch.mean(
            torch.tensor(
                [
                    self.prob(trajectory[0, :], trajectory[1, :])
                    * advantage(trajectory[2, :])
                    for trajectory in trajectories
                ]
            )
        )

    def train(self, trajectories):
        loss = self._compute_loss(trajectories)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def prob(self, observation, action):
        """
        Return probability of taking an action given an observation.
        """
        return self.model(observation)[action]

    def act(self, observation, info):
        return torch.distributions.Categorical(self.model(observation)).sample().item()
