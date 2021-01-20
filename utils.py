import numpy as np
import matplotlib.pyplot as plt
from module import Module

def create_C(y, l, m):
    """
    C[i][j] =  example j has label i ? 1 : 0
    :param m:
    :param y: vector of labels size (m, ) where yi belongs to {0...l-1}
    :return: C size (l, m) where l is the num of classes, m is the num of examples in batch.
    """
    C = np.zeros((l, m))
    for i, cl in enumerate(y):
        C[cl, i] = 1
    return C


def de_convert_C(C):
    """
    :param C: size (l, m) where l is the number of classes and m is the batch size
    :return: Compress the one-hot matrix to a 1-dimension vector of indices.
    """
    pred = lambda x: x >= 1
    y = []
    for col in range(C.shape[1]):
        y.append(next(i for i, x in enumerate(C[:, col]) if pred(x)))
    return y


def jack_test(X, y, nn: Module, loss: Module):
    # gradient for all the net

    logits = nn.forward(X)
    # loss_grad = (np.array([1]*1).reshape(1, 1), None, None)
    loss_value = loss.forward(logits, y)
    loss_grad = loss.backward()
    _, last_grad = nn.backward(loss_grad)
    dx, dw, db = last_grad
    d_for_x = np.random.dirichlet(np.ones(X.shape[0]), size=1).T  # (r, 1)
    e_0 = 2
    epsilons = []
    linear = []
    quad = []
    for i in range(10):
        e_now = ((0.5) ** i) * e_0
        epsilons.append(e_now)
        v = d_for_x * e_now
        f_x = loss_value
        f_x_pert = loss.forward(nn.forward(X + v), y)
        jack = dx.T @ v
        linear.append(np.abs(f_x_pert - f_x))
        quad.append(np.abs(f_x_pert - f_x - jack.item()))

    plt.xlabel = 'epsilon'
    plt.ylabel = 'error'
    plt.plot(epsilons, linear, label='linear-x')
    plt.legend()
    plt.show()
    plt.plot(epsilons, quad, label='quad-x')
    plt.legend()
    plt.show()
    plt.plot(epsilons, [x / y for x, y in zip(quad, linear)], label='quad/linear')
    plt.legend()
    plt.show()


def grad_check(f, X, W, Y, b, dx, dw, db):
    d_for_x = np.random.dirichlet(np.ones(X.shape[0]), size=1).T  # (r, 1)
    d_for_W = np.random.dirichlet(np.ones(W.shape).flatten(), size=1).reshape(W.shape)  # (r, l, 1)
    d_for_b = np.random.dirichlet(np.ones(b.shape), size=1).T  # (r, 1)
    linear1 = []
    quad1 = []
    linear2 = []
    quad2 = []
    linear3 = []
    quad3 = []
    epsilons = []
    e_0 = 2
    for i in range(10):
        e_now = ((0.5) ** i) * e_0
        epsilons.append(e_now)
        res_X = linear_calc(W, X, b)  # (l, m)
        X_pert = linear_calc(W, X + d_for_x * e_now, b)  # (l, m)
        W_pert = linear_calc(W + d_for_W * e_now, X, b)  # (l, m)
        b_pert = linear_calc(W, X, b + d_for_b * e_now)  # (l, m)
        f_x = f(X_pert, Y)
        f_w = f(W_pert, Y)
        f_b = f(b_pert, Y)
        a2 = f(res_X, Y)
        grad_x = e_now * (d_for_x.T @ dx).item()
        grad_W = e_now * (d_for_W.flatten().T @ dw.flatten()).item()
        grad_b = e_now * (d_for_b.T @ db).item()
        linear1.append(np.abs(f_x - a2))
        quad1.append(np.abs(f_x - a2 - grad_x))
        linear2.append(np.abs(f_w - a2))
        quad2.append(np.abs(f_w - a2 - grad_W))
        linear3.append(np.abs(f_b - a2))
        quad3.append(np.abs(f_b - a2 - grad_b))

    plt.plot(epsilons, linear1, label='linear-x')
    plt.plot(epsilons, linear2, label='linear-w')
    plt.plot(epsilons, linear3, label='linear-x')
    plt.legend()
    plt.show()
    plt.plot(epsilons, quad1, label='quad-x')
    plt.plot(epsilons, quad2, label='quad-w')
    plt.plot(epsilons, quad3, label='quad-b')
    plt.legend()
    plt.show()


def linear_calc(W, X, b):
    """
    Linear layer
    :param W: (r, l(optionally))
    :param X: (r, m)
    :param b: (l, 1)
    :return:
    """
    return W.T @ X + b  # (l, r) @ (r , m) => (l,m) + (l, 1) => (l, m)
