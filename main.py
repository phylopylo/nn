from __future__ import annotations
from typing import List
import numpy as np
from numpy import e

def testInference(debug=False):
    layers = [2, 8, 1]

    w, b = _initWeights(layers)

    if debug:
        _printWeights(w)
        print()

    x = np.random.uniform(size=layers[0])
    y = np.random.uniform()

    output = _forwardPass(x, w, b)

    loss = _lossFunction(output[-1], y)

def _initWeights(layers, random=True):
    """ Initialize weights and biases on a uniform distribution. """
    w, b = [], []
    if random:
        for idx in range(len(layers) - 1):
            w.append(np.random.uniform(size=(layers[idx], layers[idx + 1])))
            b.append(np.random.uniform(size=(layers[idx + 1])))
    else:
        for idx in range(len(layers) - 1):
            w.append(np.zeros((layers[idx], layers[idx + 1])))
            b.append(np.zeros((layers[idx + 1])))
    
    return w, b

def _printWeights(w: List[np.ndarray]):
    print("current weights dims: ", [wL.shape for wL in w])

def sigmoid(x: np.ndarray | float):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x: np.ndarray | float):
    return x * (1 - x)

def ReLU(x: np.ndarray | float):
    return np.maximum(0, x)

def ReLUDerivative(x: np.ndarray | float):
    return x > 0

_activation = sigmoid
_activationDerivative = sigmoidDerivative

def _forwardPass(x: np.ndarray, w: List[np.ndarray], b: List[np.ndarray]):
    """
    inference mode
    wL is weights for a layer.
    bL is biases for a layer.
    """
    outputs = []

    for wL, bL in zip(w, b):
        x = _activation(x @ wL + bL)
        outputs.append(x)

    return outputs

def _lossFunction(y_pred: np.ndarray, y_true: np.ndarray):
    """ MSE """
    return ((y_pred - y_true)**2).mean()

def _gradient(w: List[np.ndarray], b: List[np.ndarray], x: np.ndarray, y: np.ndarray):
    """ Calculate Gradients """

    outputs = _forwardPass(x, w, b)
    w_delta = [np.zeros_like(wL) for wL in w]
    b_delta = [np.zeros_like(bL) for bL in b]

    error = outputs[-1].reshape(max(outputs[-1].shape)) - y
    gradient = _activationDerivative(outputs[-1]).reshape(max(outputs[-1].shape))
    delta = (error * gradient).reshape(outputs[-1].shape)

    w_delta[-1] = delta * outputs[-2]
    b_delta[-1] = delta

    error = delta @ w[-1].T
    gradient = _activationDerivative(outputs[-2])
    delta = error * gradient

    delta = delta.reshape(max(outputs[-1].shape), 8, 1)
    x = x.reshape(max(outputs[-1].shape), 1, 2)

    w_delta[-2] = delta * x
    w_delta[-2] = w_delta[-2].reshape(max(outputs[-1].shape), 2, 8)

    b_delta[-2] = delta

    return w_delta, b_delta

def _updateWeights(w: List[np.ndarray], b: List[np.ndarray], w_delta: List[np.ndarray], b_delta: List[np.ndarray], lr=0.01):
    assert all([wL.shape == wDL.shape for _, (wL, wDL) in enumerate(zip(w, w_delta))])
    assert all([bL.shape == bDL.shape for _, (bL, bDL) in enumerate(zip(b, b_delta))])

    for i in range(len(w)):
        if w[i].shape != w_delta[i].shape:
            w_delta[i] = w_delta[i].reshape(w[i].shape)
        w[i] -= lr * w_delta[i]
        b[i] -= lr * b_delta[i]

    return w, b

if "__main__" == __name__:
    testInference()
    np.random.seed(1337)
    layers = [2, 8, 1]

    w, b = _initWeights(layers)

    batch_size = 1000
    epochs = 100
    lr = 0.001

    x = np.random.choice([0, 1], size=(batch_size * epochs, 2))
    y = np.bitwise_xor.reduce(x, axis=1)

    for epoch in range(epochs):
        loss = 0
        accuracy = 0

        w_delta_cum = [np.zeros_like(wL) for wL in w]
        b_delta_cum = [np.zeros_like(bL) for bL in b]

        xB = x[epoch : epoch + batch_size]
        yB = y[epoch : epoch + batch_size]

        w_delta, b_delta = _gradient(w, b, xB, yB)

        for i in range(len(w_delta_cum)):
            b_delta_sum = np.sum(b_delta[i], axis=0)
            w_delta_sum = np.sum(w_delta[i], axis=0)

            if any([dim == 1 for dim in w_delta_cum[i].shape]):
                w_delta_cum[i] = w_delta_cum[i].flatten()

            w_delta_cum[i] += w_delta_sum
            b_delta_cum[i] += b_delta_sum.reshape(max(b_delta_sum.shape))

        for i in range(len(w)):
            w_delta_cum[i] = w_delta_cum[i].reshape(w[i].shape)
            b_delta_cum[i] = b_delta_cum[i].reshape(b[i].shape)
        
        w, b = _updateWeights(w, b, w_delta_cum, b_delta_cum, lr)

        outputs = _forwardPass(xB, w, b)
        loss += _lossFunction(outputs[-1], yB)
        accuracy += np.sum((outputs[-1] > 0.5).flatten() == yB)
        if epoch % 10 == 0:
            print(
                "epoch =",
                epoch,
                "loss =",
                loss,
                "accuracy =",
                accuracy / batch_size,
            )


    """
    TEST MODEL
    """
    test_size = 1000

    x = np.random.choice([0, 1], size=(test_size, 2))
    y = np.bitwise_xor.reduce(x, axis=1)
    acc = 0
    for i in range(test_size):
        acc += (_forwardPass(x[i], w, b)[-1][0] > 0.5).flatten() == y[i]

    print(acc / test_size)