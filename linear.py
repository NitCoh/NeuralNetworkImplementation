import numpy as np
from module import Module


class Linear(Module):
    """
    Linear layer, including activation.
    function F: R^n => R^p
    """

    def __init__(self, in_dim, out_dim, activation: Module = None):
        """
        W size (n, p)
        b size (p,)
        :param in_dim = n
        :param out_dim = p
        """
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.random.randn(out_dim, 1)
        self.activation = activation
        self.X = None
        self.forward_res = None

    def forward(self, X):
        """
        Linear layer
        :param X: (n, m), n - in channels of the layer, m - batch-size
        :return: return tensor size (p, m), p - out_dim, m - batch size
        """
        self.X = X
        self.forward_res = self.W.T @ X + self.b  # (p,n) * (n, m) => (p, m)
        return self.activation.forward(self.forward_res) if self.activation is not None else self.forward_res

    def backward(self, v):
        """
        Computing the gradient of the layer multiplied by v
        Restricted to "forward" first.
        :param dx = v[0]: size (p, m), where: p - out dim of the layer , m - batch size
        :param dw = v[1]: size (n, m), where: n - in dim of the layer , m - batch size
        :param db = v[2]: size (p, 1), where: p - out dim of the layer
        :return: dx, dw, db
        """
        # Our forward definition is W.T * X + b.
        m = v[0].shape[1]
        res = self.activation.backward(self.forward_res) if self.activation is not None else 1
        res = res * v[0]  # element-wise, res size (p, m)
        dx = self.W @ res  # (n, p) x (p, m) => (n, m)
        dw = self.X @ res.T  # (n, m) x (m, p) => (n, p)
        db = (np.sum(res, axis=1) / m).reshape(len(self.b), 1)  # average over all batch
        if self.activation is None:
            db = (np.sum(res, axis=1)).reshape(len(self.b), 1)
        return dx, dw, db

    def parameters(self):
        """
        Returns all the parameters in the layer
        :return:
        """
        return [self.W, self.b]
