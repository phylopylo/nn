from __future__ import annotations
from typing import List
import numpy as np
from numpy import e

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

def _initWeights(layers):
    """ Initialize weights and biases on a uniform distribution. """
    w, b = [], []

    for idx in range(len(layers) - 1):
        w.append(np.random.uniform(size=(layers[idx], layers[idx + 1])))
        b.append(np.random.uniform(size=(layers[idx + 1])))
    
    return w, b

def _printWeights(w: List[np.ndarray]):
    print("current weights dims: ", [wL.shape for wL in w])

def _forwardPass(x: np.ndarray, w: List[np.ndarray], b: List[np.ndarray]):
    """
    inference mode
    wL is weights for a layer.
    bL is biases for a layer.
    """
    assert x.shape[0] == w[0].shape[0]

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

    error = outputs[-1] - y
    gradient = _activationDerivative(outputs[-1])
    delta = error * gradient

    w_delta[-1] = np.outer(outputs[-2], delta)
    b_delta[-1] = delta

    for i in range(len(w) - 2, -1, -1):
        error = delta @ w[i + 1].T
        gradient = _activationDerivative(outputs[i])
        delta = error * gradient

        w_delta[i] = np.outer(x if i == 0 else outputs[i - 1], delta)
        b_delta[i] = delta

    return w_delta, b_delta

def updateWeights(w, b, w_delta, b_delta, lr=0.01):
    # for each layer, update the weights and biases with the gradients
    for i in range(len(w)):
        w[i] -= lr * w_delta[i]
        b[i] -= lr * b_delta[i]

    return w, b


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

if "__main__" == __name__:
    testInference()
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
        w_delta_cum = []
        b_delta_cum = []
        for batch in range(batch_size):
            xB = x[epoch + batch]
            yB = y[epoch + batch]
            w_delta, b_delta = _gradient(w, b, xB, yB)

            if w_delta_cum == []:
                w_delta_cum = w_delta
                b_delta_cum = b_delta
            else:
                for i in range(len(w_delta_cum)):
                    w_delta_cum[i] += w_delta[i]
                    b_delta_cum[i] += b_delta[i]

            outputs = _forwardPass(xB, w, b)
            loss += _lossFunction(outputs[-1], yB)
            accuracy += ((outputs[-1][0] > 0.499) == yB)

        w, b = updateWeights(w, b, w_delta_cum, b_delta_cum, lr)
        print("epoch=", epoch, "loss=", loss / batch_size, "accuracy=", accuracy / batch_size)

    """
    TEST MODEL
    """

    x = np.random.choice([0, 1], size=(1000, 2))
    y = np.bitwise_xor.reduce(x, axis=1)
    acc = 0
    for i in range(1000):
        acc += (_forwardPass(x[i], w, b)[-1][0] > 0.499) == y[i]

    print(acc / 1000)
