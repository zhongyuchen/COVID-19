import os
import numpy as np
from PIL import Image
import random
from dataset import Dataset
import pickle


def read_data(path):
    filenames = os.listdir(path)
    filenames = sorted(filenames)
    imgs = []
    for i, filename in enumerate(filenames):
        img=Image.open(os.path.join(path, filename))
        img=img.convert("L")
        img = img.resize((128, 128))
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
    
    train_positive = positive[0:int(pl*0.8)]
    
    test_positive = positive[int(pl*0.8):]
    test_negative = negative[int(nl*0.8):]

    return np.array(train_positive), np.array(test_positive), np.array(test_negative)


def get_dataset():
    train, tpos, tneg = get_data()

    data_path = './data'
    pickle.dump(train, open(os.path.join(data_path, 'svm_train.pkl'), 'wb'))
    pickle.dump(tpos, open(os.path.join(data_path, 'svm_test_positive.pkl'), 'wb'))
    pickle.dump(tneg, open(os.path.join(data_path, 'svm_test_negative.pkl'), 'wb'))


if __name__ == '__main__':
    get_dataset()

