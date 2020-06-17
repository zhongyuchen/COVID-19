import os
import numpy as np
from PIL import Image
import random
from dataset import Dataset
import pickle
import cv2


def read_data(path):
    filenames = os.listdir(path)
    filenames = sorted(filenames)
    imgs = []
    for i, filename in enumerate(filenames):
        img=cv2.imread(os.path.join(path, filename))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (128, 128))
        img=np.array(img)
        imgs.append(img)
    print(len(imgs))
    return imgs


def get_data():
    positive = read_data('./data/COVID')
    negative = read_data('./data/non-COVID')

    random.shuffle(positive)
    random.shuffle(negative)

    pl, nl = len(positive), len(negative)
    train_positive = positive[0:int(pl*0.64)]
    dev_positive = positive[int(pl*0.64):int(pl*0.8)]
    test_positive = positive[int(pl*0.8):]

    train_negative = negative[0:int(nl*0.64)]
    dev_negative = negative[int(nl*0.64):int(nl*0.8)]
    test_negative = negative[int(nl*0.8):]

    train = train_positive + train_negative
    train_y = [1] * len(train_positive) + [0] * len(train_negative)

    dev = dev_positive + dev_negative
    dev_y = [1] * len(dev_positive) + [0] * len(dev_negative)

    test = test_positive + test_negative
    test_y = [1] * len(test_positive) + [0] * len(test_negative)

    train = np.array(train)
    train_y = np.array(train_y)

    dev = np.array(dev)
    dev_y = np.array(dev_y)

    test = np.array(test)
    test_y = np.array(test_y)

    return train, train_y, dev, dev_y, test, test_y


def get_dataset():
    train, train_y, dev, dev_y, test, test_y = get_data()
    train = Dataset(train, train_y)
    dev = Dataset(dev, dev_y)
    test = Dataset(test, test_y)

    data_path = './data'
    pickle.dump(train, open(os.path.join(data_path, 'hist_train.pkl'), 'wb'))
    pickle.dump(dev, open(os.path.join(data_path, 'hist_dev.pkl'), 'wb'))
    pickle.dump(test, open(os.path.join(data_path, 'hist_test.pkl'), 'wb'))


if __name__ == '__main__':
    get_dataset()

