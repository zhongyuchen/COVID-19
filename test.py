import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN, VGG
import configparser
from dataset import Dataset
import os
from tensorboardx import SummaryWriter
import pickle
from train import test


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model_choice = args.model
    # setting
    torch.manual_seed(11)
    device = torch.device(0)
    # data loader
    data_path = './data'
    train_data = pickle.load(open(os.path.join(data_path, 'train.pkl'), 'rb'))
    dev_data = pickle.load(open(os.path.join(data_path, 'dev.pkl'), 'rb'))
    print('dataset', len(train_data), len(dev_data))
    kwargs = {'num_workers': 32, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dev_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    lr = 0.001
    # model setting
    Model = {
        'cnn': CNN,
        'vgg': VGG
    }
    model = Model[model_choice]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metric = nn.CrossEntropyLoss().to(device)
    acc = test(model, device, test_loader, metric)
    print(acc)
