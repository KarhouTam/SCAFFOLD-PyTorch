import os
import numpy as np
import pickle
import torch
from path import Path
from argparse import ArgumentParser
from torch.utils.data import Dataset

current_dir = Path(__file__).parent.abspath()


class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.targets = y

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def preprocess(args):
    # All codes below are modified from https://github.com/litian96/FedProx/tree/master/data
    dimension = 60
    NUM_CLASS = 10

    samples_per_user = (
        np.random.lognormal(4, 2, args.client_num_in_total).astype(int) + 50
    )
    W_global = np.zeros((dimension, NUM_CLASS))
    b_global = np.zeros(NUM_CLASS)
    X_split = [[] for _ in range(args.client_num_in_total)]
    y_split = [[] for _ in range(args.client_num_in_total)]

    mean_b = mean_W = np.random.normal(0, args.gamma, args.client_num_in_total)
    B = np.random.normal(0, args.beta, args.client_num_in_total)
    mean_x = np.zeros((args.client_num_in_total, dimension))
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(args.client_num_in_total):
        if args.iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        print(mean_x[i])

    if args.iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1, NUM_CLASS)

    for i in range(args.client_num_in_total):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)

        if args.iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = torch.tensor(xx, dtype=torch.float)
        y_split[i] = torch.tensor(yy, dtype=torch.int64)

        print("{}-th users has {} examples".format(i, len(y_split[i])))

    if os.path.isdir(current_dir / "pickles"):
        os.system("rm -rf {}/pickles".format(current_dir))
    os.mkdir(current_dir / "pickles")
    for i, (x, y) in enumerate(zip(X_split, y_split)):
        with open("{}/pickles/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump(SyntheticDataset(x, y), file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--iid", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
