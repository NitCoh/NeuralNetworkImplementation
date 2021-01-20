from utils import *
from modules import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from DataLoader import *
import statistics
from neural_network import NeuralNetwork


class SGDOptimizer:
    def __init__(self, params):
        self.params = params

    def update(self, lr, grads):
        """
        Updates the given parameters.
        :param lr:
        :param grads:
        :return:
        """
        for param, grad in zip(self.params, grads):
            param -= lr * grad

    def optimize(self, f: NeuralNetwork, loss: Module, data, epochs=250, lr=1e-01, plot=False, dl: DataLoader = None,
                 calc_acc=False):
        """
        Optimize the given module with a given loss.
        Prioritize dataloader over data.
        :param f:
        :param loss:
        :param data:
        :param epochs:
        :param lr:
        :param plot:
        :param dl:
        :return:
        """
        data = dl or data
        loss_scores = []
        accs = []
        for epoch in range(epochs):
            epoch_loss = []
            if dl:
                dl.reshuffle()
            for X, y in data:  # mini-batch
                # if epoch == 1000:
                #    lr = lr * 1e-1
                logits = f.forward(X)
                loss_value = loss.forward(logits, y)  # loss
                epoch_loss.append(loss_value)
                grads, _ = f.backward(loss.backward())
                self.update(lr, grads)

            if calc_acc:
                y_hat = f.predict(dl.m_Xtrain)
                y_true = np.argmax(dl.m_Ytrain, axis=0)
                accs.append((y_hat == y_true).mean())

            avg_loss = np.average(epoch_loss)
            print(f'Epoch {epoch} avg_loss: {avg_loss}, std: {statistics.stdev(epoch_loss)}')
            loss_scores.append(avg_loss)

        if plot:
            SGDOptimizer.plot(list(range(epochs)), loss_scores, label="Loss", ylabel="loss_scores", xlabel="epochs")
            if calc_acc:
                SGDOptimizer.plot(list(range(epochs)), accs, label="Accuracy", ylabel="Acc", xlabel="epochs")

    @staticmethod
    def plot(x, y, label, ylabel, xlabel="epochs"):
        """
        Plot the training loss graph
        :param epochs: type list. (x axis)
        :param loss_scores: Loss values aggregated during the training (y axis)
        :return:
        """
        plt.plot(x, y, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
