from __future__ import annotations
from typing import List, TypedDict
import numpy as np
from numpy import e

def sigmoid(x: np.array | float):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x: np.array | float):
    tmp = [_activation(xT) for xT in x]
    return [t * (1 - t) for t in tmp]

def ReLU(x: np.array | float):
    return np.maximum(0, x)

def ReLUDerivative(x: np.array | float):
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

def _printWeights(w: List[np.array]):
    print("current weights dims: ", [wL.shape for wL in w])

def _forwardPass(x: np.array, w: List[np.array], b: List[np.array]):
    """
    inference mode
    wL is weights for a layer.
    bL is biases for a layer.
    """
    assert x.shape[0] == w[0].shape[0]
    assert len(w) == len(b)
    assert [wL.shape == bL.shape for _, (wL, bL) in enumerate(zip(w, b))]

    outputs = [np.zeros(w[i].shape) for i in range(len(w))]

    for i, (wL, bL) in enumerate(zip(w, b)):
        x = _activation(x @ wL + bL)
        outputs[i] = x

    return outputs

def _lossFunction(y_pred: np.array, y_true: np.array):
    """ MSE """
    return ((y_pred - y_true)**2).mean()

def _gradient(w: np.array, b: np.array, x: np.array, y: np.array):
    """ Calculate Gradients """

    outputs = _forwardPass(x, w, b)
    w_delta = []

    for layer in reversed(range(len(w))):
        if layer == len(w) - 1:
            error = _lossFunction(outputs[layer], y)
            gradient = sigmoidDerivative(outputs[layer])
            w_delta.append(error * gradient[0])
        else:
            for j in range(len(w[layer])):
                error = 0
                for neuron in range(len(w[layer + 1])):
                    error += w[layer + 1][neuron] * w_delta[layer]
                gradient = sum(sigmoidDerivative(outputs[layer]))
                w_delta.append(error * gradient)

    print(w_delta)
    w_delta = list(reversed(w_delta))

    lr = 0.01

    w = [wL - lr * w_deltaL for wL, w_deltaL in enumerate(zip(w, w_delta))]

    return w

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

    x = np.random.choice([0, 1], size=(1000, 2))
    y = np.bitwise_xor.reduce(x, axis=1)

    batches = 100
    epochs = 10

    for i in range(epochs * batches):
        w = _gradient(w, b, x[i], y[i])
        print("epoch=", epoch, "loss=", _lossFunction(xB, yB))

    #for epoch in range(epochs):
    #    xB = x[epoch:epoch * batches + batches]
    #    yB = y[epoch:epoch * batches + batches]
    #    w = _gradient(w, b, xB, yB)
    #    print("epoch=", epoch, "loss=", _lossFunction(xB, yB))






