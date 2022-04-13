from torch import nn


class CNN_MNIST(nn.Module):
    def __init__(self) -> None:
        super(CNN_MNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "mnist")


class MLP_MNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_MNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "mnist")


class MLP_SYNTHETIC(nn.Module):
    def __init__(self) -> None:
        super(MLP_SYNTHETIC, self).__init__()
        self.net = nn.Linear(60, 10)

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "synthetic")


class MLP_FEMNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_FEMNIST, self).__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("mlp", "femnist")


class CNN_FEMNIST(nn.Module):
    def __init__(self) -> None:
        super(CNN_FEMNIST, self).__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "femnist")


class CNN_CIFAR(nn.Module):
    def __init__(self) -> None:
        super(CNN_CIFAR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def info(self):
        return ("cnn", "cifar")

