import os
import pickle
from path import Path
from torch.utils.data import DataLoader
from .preprocess import CIFARDataset

current_dir = Path(__file__).parent.abspath()


def get_cifar(client_id, batch_size):
    if os.path.isdir(current_dir / "pickles") is False:
        raise RuntimeError(
            "Please run data/cifar/preprocess.py to generate data first."
        )
    with open("{}/pickles/client_{}.pkl".format(current_dir, client_id), "rb") as file:
        trainset, testset = pickle.load(file)

    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size, shuffle=True)

    return trainloader, testloader

