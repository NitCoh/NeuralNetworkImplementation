from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from DataLoader import *

def sgd(f, W, b, data, batch_size=None, epochs=50, lr=1e-0001, plot=False, dl: DataLoader = None):
    data = dl or data
    loss = []
    for epoch in range(epochs):
            epoch_loss = []
            if dl:
                dl.read_data()
                dl.reshuffle()
            for X, y in data:
                # if epoch % 100 == 0:
                #     lr = lr * 1e-1
                mini_batch_loss = f(linear_calc(W, X, b), y)  # soft-max objective loss.
                epoch_loss.append(mini_batch_loss)
                dw, _, db = grads_softmax_objects(X, y, W, b)
                W = W - lr * dw
                b = b - lr * db
            loss.append(np.average(epoch_loss))

    if plot:
        plt.plot(list(range(epochs)), loss, label='loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()



def softmax(X):
    """
    Calculate soft-max on a given numpy array
    :param X:
    :return: softmax matrix/vector
    """
    X_exp = np.exp(X - np.amax(X, 0, keepdims=True))  # numerical stability
    return X_exp / X_exp.sum(0, keepdims=True)  # broadcasting


def softmax_objective(x, y):
    """
    :param X: mini-batch size (r, m): where m=num of examples, r = input size
    :param y: vector of labels size (m, ) where yi belongs to {0...l-1}
    :param W: matrix of weights from the soft-max layer size (r, l)
    :param b: bias term size (l, 1)
    :return: the objective softmax value
    """
    # x = X.T @ W + b.T
    m = len(y)
    l_s_X = np.log(softmax(x))  # (r, m).T x (r, l) = (m, l)
    # l_s_X = np.log(softmax(W.T @ X + b))  # (l, r) x (r, m) = (l, m)
    return - sum([l_s_X[i, cl] for i, cl in enumerate(y)]) / m

def grads_softmax_objects(X, y, W, b):
    """

    :param X: size (r, m)
    :param y: size (m, 1)
    :param W: size (r, l)
    :param b: size (l, 1)
    :param l:
    :param m:
    :return: tuple (gradient_by_W, gradient_by_bias)
    """

    C = create_C(y, len(b), len(y))
    y_tag = softmax(linear_calc(W, X, b))  # shape (l, m)

    m = len(y)

    ds = y_tag - C  # (l, m) - (l, m)

    db = (np.sum(y_tag - C, axis=1) / m).reshape(len(b), 1)  # y_tag = (l, m) , y = (m, 1), db = (l, 1)
    dx = np.dot(W, ds) * (1 / m)  # (r, l) x (l, m) => (r, m)
    dw = np.dot(X, ds.T) * (1 / m)  # (r, m) x (m, l) => (r, l)

    return dw, dx, db
