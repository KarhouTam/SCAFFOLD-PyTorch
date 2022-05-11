# SCAFFOLD-PyTorch

This repo is the implementation of [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)

For simulating Non-I.I.D scenario, the dataset is split by labels and each client has only **two** classes of data.

# Requirements

path~=16.4.0

torch~=1.10.2

numpy~=1.21.2

fedlab~=1.1.4

torchvision~=0.11.3

tqdm~=4.62.3

```python
pip install -r requirements.txt
```

# Preprocess dataset
  
```python
cd data/DATASET; python preprocess.py
```
The way of preprocessing is adjustable, more details in each dataset folder's `preprocess.py`.

# Run the experiment

Before run the experiment, please make sure that the dataset is downloaded and preprocessed already.

It’s so simple.🤪

```python
python main.py
```



## Hyperparameters

`--comms_round`: Num of communication rounds. Default: `100`

`--dataset`: Name of experiment dataset. Default: `cifar`

`--client_num_per_round`: Num of clients that participating training at each communication round. Default: `5`

`--test_round`: Num of round for final evaluation. Default: `1`

`--local_lr`: Learning rate for client local model updating. Default: `0.05`

`--batch_size`: Batch size of client local dataset.

`--global_lr`: Learning rate for server model updating. Default: `1.0`

`--cuda`: `True` for using GPU. Default: `True`

`--epochs`: Num of local epochs in client training. Default: `5`

`--model`: Structure of model. Must be `mlp` or `cnn`. Default: `cnn`

`--seed`: Random seed for init model parameters and selected clients.



# Result

FedAvg's result are from https://github.com/KarhouTam/Federated-Averaging-PyTorch

| Algorithm | Global Loss | Localized Loss | Global Acc | Localized Acc |
| --------- | ----------- | -------------- | ---------- | ------------- |
| FedAvg    | `10.5450`   | `4.6780`       | `19.40%`   | `52.60%`      |
| SCAFFOLD  | `10.6327`   | `3.6109`       | `20.00%`   | `58.80%`      |

Localization means the model is trained for 10 local epochs additionally at the final evaluation phase, which is for adapting client’s local dataset.

