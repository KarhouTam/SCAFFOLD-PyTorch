from .mnist import get_mnist
from .synthetic import get_synthetic
from .cifar import get_cifar


def get_dataloader(client_id, dataset, batch_size):
    if dataset == "mnist":
        return get_mnist(client_id, batch_size)
    elif dataset == "cifar":
        return get_cifar(client_id, batch_size)
    elif dataset == "synthetic":
        return get_synthetic(client_id, batch_size)
    else:
        raise NotImplementedError(
            'Dataset "{}" is not supported. Please switch to mnist, cifar or synthetic'.format(
                dataset
            )
        )

