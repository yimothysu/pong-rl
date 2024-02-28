import torch


def preprocess(observation):
    """
    Takes in tensor of shape (height, width, channels) = (210, 160, 3)
    Converts background pixels to black and the paddles / ball pixels to white
    Returns processed tensor of shape (80, 80)
    """
    out = torch.from_numpy(observation).to(torch.float32) / 255.0
    out = out[35:195, :, :]
    out = out[::2, ::2, 0]
    # Set background to 0
    out[out == out.max()] = 0.0
    # Set paddles and ball to 1
    out[out > 0] = 1.0
    return out

# TODO: maybe incorporate this into preprocess with conditionals 
def preprocess_batch(observations_batch):
    """
    Takes in tensor of shape (batch_size, height, width, channels) = (N, 210, 160, 3)
    Converts background pixels to black and the paddles / ball pixels to white
    Returns processed tensor of shape (N, 80, 80)
    """
    # scale pixel values to [0, 1]
    observations_batch = torch.from_numpy(observations_batch).to(torch.float32) / 255.0
    
    # crop rows to only include the play area 
    observations_batch = observations_batch[:, 35:195, :, :]
    
    # downsample by factor of 2 by skipping every other row/col and only use one color channel
    observations_batch = observations_batch[:, ::2, ::2, 0]
    
    # background (max values) to 0
    max_val = observations_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    observations_batch[observations_batch == max_val] = 0.0
    
    # set the paddles and ball to 1
    observations_batch[observations_batch > 0] = 1.0
    
    return observations_batch