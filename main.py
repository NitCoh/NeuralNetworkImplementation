from modules import *
from neural_network import NeuralNetwork
from sgd import SGDOptimizer
import argparse
from DataLoader import DataLoader
from first_tasks import *


PREFIX = './data'
def train(path, pre_NN=None, dl=None):
    if pre_NN is None and dl is None:
        dl = DataLoader(path, 64, True)
        input_dim = len(dl.m_Xtrain)  # input dimension
        output_dim = len(dl.m_Ytest)  # num of classes
        nn = NeuralNetwork(input_dim=input_dim, output_dim=output_dim)
    else:
        nn = pre_NN
        dl = dl
    optim = SGDOptimizer(nn.parameters())
    optim.optimize(nn, SoftmaxCrossEntropyLoss(), None, plot=True, dl=dl, calc_acc=True)
    y_test_hat = nn.predict(dl.m_Xtest)
    y_test_true = np.argmax(dl.m_Ytest, axis=0)
    acc = (y_test_hat == y_test_true).mean()
    print(f'Final Accuracy on Test: {acc}')


def mini_train(X, y):
    nn = NeuralNetwork()
    optim = SGDOptimizer(nn.parameters())
    optim.optimize(nn, SoftmaxCrossEntropyLoss(), [[X, y]], plot=True)


def task1():
    X = np.array([[1, 2, 3, 4]]).T  # (r, m)
    Y = [0]
    b = np.expand_dims(np.array([0, 1]), axis=1)
    W = np.random.rand(4, 2)
    dw, dx, db = grads_softmax_objects(X, Y, W, b)
    grad_check(softmax_objective, X, W, Y, b, dx, dw, db)


def task3():
    X = np.array([[1, 2, 3, 4]]).T  # (r, m)
    Y = [0]
    b = np.expand_dims(np.array([0, 1]), axis=1)
    W = np.random.rand(4, 2)
    sgd(softmax_objective, W, b, [[X, Y]], plot=True)
    data_paths = [f"{PREFIX}/GMMData.mat", f"{PREFIX}/PeaksData.mat"]

    for dp in data_paths:
        dl = DataLoader(dp, 64)
        input_dim = len(dl.m_Xtrain)  # input dimension
        output_dim = len(dl.m_Ytest)  # num of classes
        nn = NeuralNetwork(use_arch=False)
        nn.add_layer(in_dim=input_dim, out_dim=output_dim, activation=None)
        train(None, pre_NN=nn, dl=dl)


def task6():
    X = np.array([[1, 2, 3, 4]]).T
    y = [0]
    Y = create_C(y, 2, X.shape[1])
    nn = NeuralNetwork(use_arch=False)
    nn.add_layer(in_dim=4, out_dim=3, activation=Tanh())
    nn.add_layer(in_dim=3, out_dim=2, activation=None)
    loss = SoftmaxCrossEntropyLoss()
    jack_test(X, Y, nn, loss)


def task7():
    data_paths = [f"{PREFIX}/GMMData.mat", f"{PREFIX}/SwissRollData.mat"]
    for data_path in data_paths:
        train(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type=str, help="Choose the task you want to run: task1 | task3 | task6 | task7")
    parser.add_argument("-path", type=str, help="Insert the absolute/relative prefix path for folder that the data is lying")
    args = parser.parse_args()

    if args.path:
        PREFIX = args.path

    tasks = {
        "task1": task1,
        "task3": task3,
        "task6": task6,
        "task7": task7
    }

    if args.task in tasks:
        tasks[args.task]()
    else:
        print(f'Non valid task chosen, only possible {tasks.keys()}')

