# SCAFFOLD: Stochastic Controlled Averaging for Federated Learning [[ArXiv]](https://arxiv.org/abs/1910.06378) 
## This repo is not maintained anymore, you can transfer to [FL-bench](https://github.com/KarhouTam/FL-bench), thanks guys!

This repo is the PyTorch implementation of SCAFFOLD.

I further implement FedAvg and FedProx for you.🤗

For simulating Non-I.I.D scenario, the dataset can be splitted based on Dirchlet distribution or assign random classes to each client.

Note that I have recently released a [benchmark of federated learning](https://github.com/KarhouTam/FL-bench) that includes this method and many ohter baselines. Welcome to check my benchmark and star it! 🤗

## Preprocess dataset
  
MNIST, EMNIST, FashionMNIST, CIFAR10, CIFAR100 are supported.

```python
python ./data/utils/run.py --dataset ${dataset}
```
The way of preprocessing is adjustable. Check `./data/utils/run.py` for more argument details
## Run the experiment

❗ Before run the experiment, please make sure that the dataset is downloaded and preprocessed already.

It’s so simple.🤪

```python
python ./src/server/${algo}.py
```

You can check `./src/config/util.py` for all hyperparameters detail.


## Result

❗NOTE: The dataset settings, hyperparameters, and model backbone in this repo are not the same as in the SCAFFOLD paper. So the result below doesn't mean anything. 

This repo is just for showing the process of SCAFFOLD.

If something wrong you find in any alogorithms' process in this repo, just let me know. 🤗 

Some stats about convergence speed are shown below.

`--dataset`: `emnist`. Splitted by Dirchlet(0.5)

`--global_epochs`: `100`

`--local_epochs`: `10`

`--client_num_in_total`: `10`

`--client_num_per_round`: `2`

`--local_lr`: `1e-2`

`--seed`: `17`


| Algo     | Epoch to 50% Acc | Epoch to 60% Acc | Epoch to 70% Acc | Epoch to 80% Acc | Test Acc |
| -------- | ---------------- | ---------------- | ---------------- | ---------------- | -------- |
| FedAvg   | 6                | 16               | 30               | 56               | 70.00%   |
| FedProx  | 12               | 14               | 30               | 56               | 66.72%   |
| SCAFFOLD | 6                | 15               | 27               | -                | 53.93%   |
