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
    """Initialize weights and biases on a uniform distribution."""
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
    outputs = []

    for wL, bL in zip(w, b):
        x = _activation(x @ wL + bL)
        outputs.append(x)

    return outputs


def _lossFunction(y_pred: np.ndarray, y_true: np.ndarray):
    """MSE"""
    return ((y_pred - y_true) ** 2).mean()


def _gradient(w: List[np.ndarray], b: List[np.ndarray], x: np.ndarray, y: np.ndarray):
    """Calculate Gradients"""

    outputs = _forwardPass(x, w, b)
    outputs[-1] = outputs[-1].reshape(max(outputs[-1].shape))

    w_delta = [np.zeros_like(wL) for wL in w]
    b_delta = [np.zeros_like(bL) for bL in b]

    # Calculate gradients for layers 2-3 (single hidden to output)
    error = outputs[-1] - y
    gradient = _activationDerivative(outputs[-1])
    delta = error * gradient
    print(error.shape)
    print(gradient.shape)
    print(delta.shape)
    print()

    w_delta[-1] = np.array([outputs[-2][i] * delta[i] for i in range(len(delta))])
    b_delta[-1] = np.ones_like(b_delta[-1]) * delta

    # Calculate gradients for layers 1-2 (input to single hidden)
    error = delta * w[-1]
    error = error.T  # we transpose so the first channel is the batch size
    gradient = _activationDerivative(outputs[-2])
    delta = error * gradient

    w_delta[-2] = outputs[-2] * delta
    b_delta[-2] = delta

    return w_delta, b_delta


def updateWeights(w, b, w_delta, b_delta, lr=0.01):
    for i in range(len(w_delta)):
        for ii in range(len(w)):
            w[ii] -= lr * w_delta[i][ii]
            b[ii] -= lr * b_delta[i][ii]

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

        w_delta_cum = [np.zeros_like(wL) for wL in w]
        b_delta_cum = [np.zeros_like(bL) for bL in b]

        xB = x[epoch : epoch + batch_size]
        yB = y[epoch : epoch + batch_size]
        w_delta, b_delta = _gradient(w, b, xB, yB)
        print([np.array(lmao).shape for lmao in w_delta])
        exit()

        print([np.array(wD).shape for wD in w_delta])
        print([np.array(wD).shape for wD in w_delta_cum])

        for i in range(len(w_delta_cum)):
            w_delta_cum[i] += w_delta[i]
            b_delta_cum[i] += b_delta[i]

        outputs = _forwardPass(xB, w, b)
        loss += _lossFunction(outputs[-1], yB)
        accuracy += (outputs[-1][0] > 0.499) == yB

        w, b = updateWeights(w, b, w_delta_cum, b_delta_cum, lr)
        print(
            "epoch=",
            epoch,
            "loss=",
            loss / batch_size,
            "accuracy=",
            accuracy / batch_size,
        )

    """
    TEST MODEL
    """

    x = np.random.choice([0, 1], size=(1000, 2))
    y = np.bitwise_xor.reduce(x, axis=1)
    acc = 0
    for i in range(1000):
        acc += (_forwardPass(x[i], w, b)[-1][0] > 0.499) == y[i]

    print(acc / 1000)
