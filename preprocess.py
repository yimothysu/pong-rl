import torch


def preprocess(observation):
    out = torch.from_numpy(observation).to(torch.float32) / 255.0
    out = out[35:195, :, :]
    return out[::2, ::2, 0]
