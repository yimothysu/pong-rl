"""
Vanilla Policy Gradient
"""

import torch
from torchtyping import TensorType


def reward_to_go(
    rewards: TensorType["N", "H"], discount_factor=0.99
) -> TensorType["N", "H"]:
    out = rewards.clone()
    for i in range(rewards.shape[0]):
        for j in range(rewards.shape[1] - 2, -1, -1):
            if out[i, j] == 0:
                out[i, j] = out[i, j + 1] * discount_factor
    return out


def baseline(trajectory_rewards: TensorType["N", "H"]):
    # TODO: Add baseline
    return 0


def compute_advantage(trajectory_rewards: TensorType["N", "H"]) -> torch.Tensor:
    """ """
    out = reward_to_go(trajectory_rewards) - baseline(trajectory_rewards)
    # Normalize rewards
    out = (out - out.mean()) / (out.std() + 1e-8)
    return out


class Model(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 256)
        self.fc2 = torch.nn.Linear(256, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class Policy:
    def __init__(self, obs_dim, act_dim):
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
        # n_observations:       (N, H, height, width) = (N, H, 80, 80)
        # n_actions, n_rewards: (N, H)
        n_observations, n_actions, n_rewards = trajectories
        advantages = compute_advantage(n_rewards)

        loss = -(
            self.log_prob(
                n_observations.reshape(
                    n_observations.shape[0], n_observations.shape[1], -1
                ),
                n_actions,
            )
            * advantages
        ).mean()
        return loss

    def _dist(self, obs):
        return torch.distributions.Categorical(logits=self.model(obs))

    def train(self, trajectories):
        loss = self._compute_loss(trajectories)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        print("Loss:", loss.item())

    def log_prob(self, obs, act):
        """
        Return probability of taking an action given an observation.
        """
        return self._dist(obs).log_prob(act)

    def act(self, obs):
        return self._dist(obs).sample().item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
