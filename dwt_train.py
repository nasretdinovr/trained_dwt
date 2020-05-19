import json
from train_utils import parse_agrs, train

if __name__ == "__main__":
    hparams = parse_agrs()
    train(hparams)

