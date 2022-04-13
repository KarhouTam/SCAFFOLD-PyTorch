from sys import path

path.append("../")

import torch
from fedlab.core.client import ClientTrainer
from fedlab.utils.serialization import SerializationTool
from tqdm import trange
from data import get_dataloader
from copy import deepcopy
from utils import evaluate
from math import ceil


class SCAFFOLDTrainer(ClientTrainer):
    def __init__(self, global_model, lr, criterion, epochs, cuda):
        super().__init__(deepcopy(global_model), cuda and torch.cuda.is_available())
        self.global_model = global_model
        self.epochs = epochs
        self.criterion = criterion
        self.device = next(iter(self.model.parameters())).device
        self.c_local = [
            torch.zeros_like(param, device=self.device)
            for param in self.model.parameters()
            if param.requires_grad
        ]
        self.lr = lr

    def train(self, client_id, global_model_parameters, c_global, dataset, batch_size):
        trainloader, _ = get_dataloader(client_id, dataset, batch_size)
        SerializationTool.deserialize_model(self.model, global_model_parameters)

        self._train(self.model, trainloader, c_global, self.epochs, client_id)
        with torch.no_grad():
            y_delta = [torch.zeros_like(param) for param in self.model.parameters()]
            c_new = deepcopy(y_delta)
            c_delta = deepcopy(
                y_delta
            )  # c_+ and c_delta both have the same shape as y_delta

            # calc model_delta (difference of model before and after training)
            for y_del, param_l, param_g in zip(
                y_delta, self.model.parameters(), self.global_model.parameters()
            ):
                y_del.data += param_l.data.detach() - param_g.data.detach()

            # update client's local control
            a = ceil(len(trainloader.dataset) / batch_size) * self.epochs * self.lr
            for c_n, c_l, c_g, diff in zip(c_new, self.c_local, c_global, y_delta):
                c_n.data += c_l.data - c_g.data + diff.data / a

            # calc control_delta
            for c_d, c_n, c_l in zip(c_delta, c_new, self.c_local):
                c_d.data += c_n.data - c_l.data

        return y_delta, c_delta

    def eval(self, client_id, global_model_parameters, c_global, dataset, batch_size):
        trainloader, testloader = get_dataloader(client_id, dataset, batch_size)
        model_4_eval = deepcopy(self.model)
        SerializationTool.deserialize_model(model_4_eval, global_model_parameters)
        # evaluate global SCAFFOLD performance
        loss_g, acc_g = evaluate(model_4_eval, testloader, self.criterion, self.device)
        # localization
        self._train(model_4_eval, trainloader, c_global, 10, client_id)
        # evaluate localized SCAFFOLD performance
        loss_l, acc_l = evaluate(model_4_eval, testloader, self.criterion, self.device)
        return loss_g, acc_g, loss_l, acc_l

    def _train(self, model, trainloader, c_global, epochs, client_id):
        model.train()
        for _ in trange(epochs, desc="client [{}]".format(client_id)):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = model(x)
                loss = self.criterion(logit, y)
                gradients = torch.autograd.grad(loss, model.parameters())
                with torch.no_grad():
                    for param, grad, c_g, c_l in zip(
                        model.parameters(), gradients, c_global, self.c_local
                    ):
                        c_g, c_l = c_g.to(self.device), c_l.to(self.device)
                        param.data = param.data - self.lr * (
                            grad.data + c_g.data - c_l.data
                        )

            self.lr *= 0.95
