import os
import pickle
from path import Path
from torch.utils.data import DataLoader, random_split
from .preprocess import SyntheticDataset


current_dir = Path(__file__).parent.abspath()


def get_synthetic(client_id, batch_size):
    if os.path.isdir(current_dir / "pickles") is False:
        # preprocess()
        raise RuntimeError(
            "Please run data/synthetic/preprocess.py to generate data first."
        )
    dataset = pickle.load(
        open("{}/pickles/client_{}.pkl".format(current_dir, client_id), "rb")
    )
    trainset, testset = random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )

    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size, shuffle=True)

    return trainloader, testloader

