from module import Module
import numpy as np

class Softmax(Module):

    def forward(self, X):
        """
        Calculate soft-max on a given numpy array
        :param X:
        :return: softmax matrix/vector
        """
        X_exp = np.exp(X - np.amax(X, 0, keepdims=True))  # numerical stability
        return X_exp / X_exp.sum(0, keepdims=True)  # broadcasting

    def backward(self, X):
        """
        :param X:
        :return:
        """
        pass


class Sigmoid(Module):

    def __init__(self):
        self.forward_res = None

    def forward(self, X):
        """
        Sigmoid element-wise
        :param X: size (r, m)
        :return:
        """
        self.forward_res = 1 / (1 + np.exp(-X))
        return self.forward_res

    def backward(self, X):
        return self.forward_res * (1-self.forward_res)


class Tanh(Module):

    def __init__(self):
        self.forward_res = None

    def forward(self, X):
        """
        Tanh element-wise
        :param X: size (r, m)
        :return:
        """
        self.forward_res = np.tanh(X)
        return np.tanh(X)

    def backward(self, X):
        return 1-(self.forward_res**2)

class ReLU(Module):

    def forward(self, X):
        """
        ReLU element-wise
        :param X:
        :return:
        """
        return X * (X > 0)

    def backward(self, X):
        """
        1) X>=0 -> dRelu/dX = 1
        2) x < 0 -> dRelu/dX = 0
        :param X:
        :return:
        """
        return 1. * (X > 0)


class SoftmaxCrossEntropyLoss(Module):
    """
    Definition of a loss is Loss(X, y)
    where X is a mini-batch of logits from the last layer.
    """

    def __init__(self):
        self.softmax = Softmax()
        self.forward_res = None
        self.C = None

    def forward(self, X, y):
        """
        Compute the loss function over m samples.
        :param y: vector of labels size (m, ) where yi belongs to {0...l-1}
        :param X: is a softmax matrix shape (l, m)
        :return:
        """
        m = len(y)
        self.forward_res = self.softmax.forward(X)
        self.C = y
        # self.C = create_C(y, *X.shape)  # (l, m)
        return - np.einsum('ij,ij', np.log(self.forward_res), y) / m

    def backward(self):
        """
        Denote qi=softmax(xi), zi = i'th logit
        grad_by_zi = qi - indicator(y=i)
        :param X: size (l, m)
        :param y: size (m, 1)
        :return: tuple
        """
        y_tag = self.forward_res  # shape (l, m)
        m = y_tag.shape[1]
        dx = (y_tag - self.C) / m  # (l, m) - (l, m)
        return dx, None, None

class CrossEntropyLoss(Module):

    def forward(self, X, y):
        """
        Compute the loss function over m samples.
        :param y: vector of labels size (m, ) where yi belongs to {0...l-1}
        :param X: is a softmax matrix shape (l, m)
        :return:
        """
        m = len(y)
        C = create_C(y, *X.shape)  # (l, m)
        return - np.einsum('ij,ij', X, C) / m

    def backward(self, v):
        return 1


