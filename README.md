# Solving Pong w/ Reinforcement Learning (Policy Gradients)

## Install

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```sh
python train.py --epochs <NUM_EPOCHS>
```

After training for 1,000 epochs, the model becomes able to outplay the opponent.

## Run

```sh
python run.py
```

To use pretrained weights, rename `model_1000.pt` to `model.pt` and run.
