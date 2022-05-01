import torch
import random
from torch.nn import CrossEntropyLoss
from scaffold import SCAFFOLDTrainer
from utils import get_args, get_model
from argparse import ArgumentParser
from tqdm import trange
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import get_best_gpu
from os import listdir

# =================== Can't not remove these modules ===================
# these modules are imported for pickles.load() deserializing properly.
from data.cifar import CIFARDataset
from data.mnist import MNISTDataset
from data.synthetic import SyntheticDataset
# ======================================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    args = get_args(parser)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.cuda and torch.cuda.is_available():
        device = get_best_gpu()
    else:
        device = torch.device("cpu")

    global_model = get_model((args.model, args.dataset)).to(device)
    c_global = [
        torch.zeros_like(param, device=device)
        for param in global_model.parameters()
        if param.requires_grad
    ]

    criterion = CrossEntropyLoss()
    client_num_in_total = len(listdir("data/{}/pickles".format(args.dataset)))
    client_indices = range(client_num_in_total)
    client_list = [
        SCAFFOLDTrainer(
            client_id=client_id,
            global_model=global_model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            lr=args.local_lr,
            criterion=criterion,
            epochs=args.epochs,
            cuda=args.cuda,
        )
        for client_id in range(client_num_in_total)
    ]

    for r in trange(args.comms_round, desc="\033[1;33mtraining epoch\033[0m"):
        # select clients
        selected_clients = random.sample(client_indices, args.client_num_per_round)
        print(
            "\033[1;34mselected clients in round [{}]: {}\033[0m".format(
                r, selected_clients
            )
        )
        global_model_param = SerializationTool.serialize_model(global_model)
        c_delta_buffer = []
        y_delta_buffer = []
        # train
        for client_id in selected_clients:
            y_delta, c_delta = client_list[client_id].train(
                global_model_param, c_global
            )
            c_delta_buffer.append(c_delta)
            y_delta_buffer.append(y_delta)

        with torch.no_grad():
            # update global model
            for y_del in y_delta_buffer:
                for param, diff in zip(global_model.parameters(), y_del):
                    param.data.add_(
                        diff.data * args.global_lr / args.client_num_per_round
                    )
            # update global_c
            for c_delta in c_delta_buffer:
                for c_g, c_d in zip(c_global, c_delta):
                    c_g.data += c_d.data / client_num_in_total

    # evaluate
    avg_loss_g = 0  # global model loss
    avg_acc_g = 0  # global model accuracy
    avg_loss_l = 0  # localized model loss
    avg_acc_l = 0  # localized model accuracy
    for r in trange(args.test_round, desc="\033[1;36mevaluating epoch\033[0m"):
        selected_clients = random.sample(client_indices, args.client_num_per_round)
        print(
            "\033[1;34mselected clients in round [{}]: {}\033[0m".format(
                r, selected_clients
            )
        )
        global_model_param = SerializationTool.serialize_model(global_model)
        for client_id in selected_clients:
            stats = client_list[client_id].eval(global_model_param, c_global)
            avg_loss_g += stats[0]
            avg_acc_g += stats[1]
            avg_loss_l += stats[2]
            avg_acc_l += stats[3]

    # display experiment results
    avg_loss_g /= args.client_num_per_round * args.test_round
    avg_acc_g /= args.client_num_per_round * args.test_round
    avg_loss_l /= args.client_num_per_round * args.test_round
    avg_acc_l /= args.client_num_per_round * args.test_round
    print("\033[1;32m---------------------- RESULTS ----------------------\033[0m")
    print("\033[1;33m Global SCAFFOLD loss: {:.4f}\033[0m".format(avg_loss_g))
    print("\033[1;33m Global SCAFFOLD accuracy: {:.2f}%\033[0m".format(avg_acc_g))
    print("\033[1;36m Localized SCAFFOLD loss: {:.4f}\033[0m".format(avg_loss_l))
    print("\033[1;36m Localized SCAFFOLD accuracy: {:.2f}%\033[0m".format(avg_acc_l))
