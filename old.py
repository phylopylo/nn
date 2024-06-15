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
    assert len(w) == len(b)

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

    print(delta.shape)
    print(outputs[-2].shape)

    w_delta[-1] = np.outer(outputs[-2], delta)
    b_delta[-1] = delta

    for i in range(len(w) - 2, -1, -1):
        error = delta @ w[i + 1].T
        gradient = _activationDerivative(outputs[i])
        delta = error * gradient

        w_delta[i] = np.outer(x if i == 0 else outputs[i - 1], delta)
        b_delta[i] = delta

    lr = 0.01
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

    x = np.random.choice([0, 1], size=(100000, 2))
    y = np.bitwise_xor.reduce(x, axis=1)

    batches = 10000
    epochs = 10

    loss = 0
    accuracy = 0

    for e in range(epochs * batches):
        if e % batches == 0:
            print("epoch=", e, "loss=", loss / batches, "accuracy=", accuracy / batches)
            loss = 0
            accuracy = 0
        xB = x[e]
        yB = y[e]
        w, b = _gradient(w, b, xB, yB)

        outputs = _forwardPass(xB, w, b)
        loss += _lossFunction(outputs[-1], yB)
        accuracy += abs(outputs[-1] - yB)
