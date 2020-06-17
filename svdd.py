import numpy as np
from sklearn.svm import OneClassSVM
import time
import pickle
import os



def svdd(rbf, nu):
    data_path = './data'
    train = pickle.load(open(os.path.join(data_path, 'svm_train.pkl'), 'rb'))
    tpos = pickle.load(open(os.path.join(data_path, 'svm_test_positive.pkl'), 'rb'))
    tneg = pickle.load(open(os.path.join(data_path, 'svm_test_negative.pkl'), 'rb'))
    # svm
    ocsvm = OneClassSVM(kernel=kernel, gamma='auto', tol=1e-3, nu=nu, shrinking=True, max_iter=-1)
    train = train.reshape(len(train), -1)
    ocsvm.fit(train)
    test = np.array(list(tpos) + list(tneg))
    test = test.reshape(len(test), -1)
    print(test.shape)
    print(len(tpos) + len(tneg))
    pred = ocsvm.predict(test)
    right = np.sum(pred[0:len(tpos)] == 1) + np.sum(pred[len(tpos):] == -1)
    print(right / len(test))


if __name__ == "__main__":

    kernel = 'rbf'
    nu = 0.1
    start_time = time.time()
    svdd(rbf=kernel, nu=nu)
    