import torch
from argparse import ArgumentParser
from models import *


def get_args(parser: ArgumentParser):
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--comms_round", type=int, default=40)
    parser.add_argument("--client_num_per_round", type=int, default=5)
    parser.add_argument("--test_round", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--global_lr", type=float, default=1.0)
    parser.add_argument("--local_lr", type=float, default=5e-2)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_model(model_info):
    _struct, _dataset = model_info
    if _dataset == "mnist":
        if _struct == "mlp":
            return MLP_MNIST()
        elif _struct == "cnn":
            return CNN_MNIST()
        else:
            raise ValueError
    elif _dataset == "cifar":
        if _struct == "cnn":
            return CNN_CIFAR()
        else:
            raise NotImplementedError
    elif _dataset == "femnist":
        if _struct == "mlp":
            return MLP_FEMNIST()
        elif _struct == "cnn":
            return CNN_FEMNIST()
        else:
            raise ValueError
    elif _dataset == "synthetic":
        if _struct == "mlp":
            return MLP_SYNTHETIC()
        else:
            raise NotImplementedError


@torch.no_grad()
def evaluate(model, testloader, criterion, gpu=None):
    model.eval()
    correct = 0
    loss = 0
    if gpu is not None:
        model = model.to(gpu)
    for x, y in testloader:
        if gpu is not None:
            x, y = x.to(gpu), y.to(gpu)

        logit = model(x)
        loss += criterion(logit, y)

        pred_y = torch.softmax(logit, -1).argmax(-1)
        correct += torch.eq(pred_y, y).int().sum()

    acc = 100.0 * (correct / len(testloader.dataset))
    return loss, acc
