import torch


def preprocess(observation):
    out = torch.from_numpy(observation).to(torch.float32) / 255.0
    out = out[35:195, :, :]
    out = out[::2, ::2, 0]
    # Set background to 0
    out[out == out.max()] = 0.0
    # Set paddles and ball to 1
    out[out > 0] = 1.0
    return out
