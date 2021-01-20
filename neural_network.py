from module import Module
from modules import *
from linear import *
import itertools
from first_tasks import softmax


class NeuralNetwork(Module):
    def __init__(self, use_arch=True, input_dim=5, output_dim=5):
        # Architecture
        if use_arch:
            self.modules = [Linear(in_dim=input_dim, out_dim=10, activation=Tanh()),
                            Linear(in_dim=10, out_dim=7, activation=Tanh()),
                            Linear(in_dim=7, out_dim=output_dim, activation=None)]
        else:
            self.modules = []


    def predict(self, X):
        """

        :param X: (l, m)
        :return:
        """
        logits = self.forward(X)
        return np.argmax(softmax(logits), axis=0)

    def forward(self, X):
        """
        Returns the final logits.
        :param X: shape (r, m) - r = input size, m = batch size
        :param y:
        :return:
        """
        res = X
        for i, module in enumerate(self.modules):
            res = module.forward(res)

        return res

    def backward(self, loss_grad):
        """Backprop implementation"""
        grads = []
        x = loss_grad
        for module in self.modules[::-1]:
            x = module.backward(x)
            dx, dw, db = x
            grads.insert(0, [dw, db])

        return NeuralNetwork.flatten(grads), x

    def parameters(self):
        """
        Returns all the parameters in the network
        :return:
        """
        return NeuralNetwork.flatten([module.parameters() for module in self.modules])

    def add_layer(self, in_dim, out_dim, activation: Module or None, i=None):
        """
        Adds layer to the i'th place in the net
        :param in_dim:
        :param out_dim:
        :param activation:
        :param i:
        :return:
        """
        i = i or len(self.modules)
        self.modules.insert(i, Linear(in_dim=in_dim, out_dim=out_dim, activation=activation))

    @staticmethod
    def flatten(lst):
        return list(itertools.chain(*lst))

