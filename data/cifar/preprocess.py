import numpy as np
import os
import pickle
import torch
from path import Path
from argparse import ArgumentParser
from fedlab.utils.dataset.slicing import noniid_slicing
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset

current_dir = Path(__file__).parent.abspath()


class CIFARDataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), subset)))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def preprocess(args):
    if os.path.isdir(current_dir / "pickles"):
        os.system("rm -rf {}/pickles".format(current_dir))
    cifar_train = CIFAR10(
        current_dir, train=True, transform=transforms.ToTensor(), download=True
    )
    cifar_test = CIFAR10(current_dir, transform=transforms.ToTensor(), train=False)
    np.random.seed(args.seed)
    train_idxs = noniid_slicing(
        cifar_train, args.client_num_in_total, args.classes * args.client_num_in_total,
    )

    # Set random seed again is for making sure numpy split trainset and testset in the same way.
    np.random.seed(args.seed)
    test_idxs = noniid_slicing(
        cifar_test, args.client_num_in_total, args.classes * args.client_num_in_total,
    )
    # Now train_idxs[i] and test_idxs[i] have the same classes.

    all_trainsets = []
    all_testsets = []

    for train_indices, test_indices in zip(train_idxs.values(), test_idxs.values()):
        all_trainsets.append(CIFARDataset([cifar_train[i] for i in train_indices]))
        all_testsets.append(CIFARDataset([cifar_test[i] for i in test_indices]))
    os.mkdir(current_dir / "pickles")
    # Store clients local trainset and testset as pickles.
    for i in range(args.client_num_in_total):
        with open("{}/pickles/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_testsets[i]), file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
