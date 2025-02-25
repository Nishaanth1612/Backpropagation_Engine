# Backpropagation (Autograd) Implementation in Python

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)


## Introduction
This project implements the backpropagation algorithm (Autograd) in Python. Backpropagation is essential for training neural networks, allowing the network to adjust weights through gradient descent to minimize the loss function. This implementation is built from scratch to help understand the underlying mechanics of backpropagation and neural networks.

## Features
- **Value class**: Represents a single scalar value and its gradient.
- **Neuron class**: Represents a single neuron in a neural network.
- **Layer class**: Represents a layer of neurons.
- **MLP class**: Represents a multi-layer perceptron neural network.

```python
class Value:
    """
    Represents a single scalar value and its gradient.

    Attributes:
        value (float): The scalar value.
        grad (float): The gradient of the value.

    Example:
        >>> x = Value(2.0)
        >>> x.value
        2.0
        >>> x.grad
        0.0
    """
    def __init__(self, value: float):
        self.value = value
        self.grad = 0.0

class Neuron:
    """
    Represents a single neuron in a neural network.

    Attributes:
        weights (list[Value]): The weights of the neuron.
        bias (Value): The bias of the neuron.
        output (Value): The output of the neuron.

    Example:
        >>> neuron = Neuron([Value(1.0), Value(2.0)], Value(3.0))
        >>> neuron.weights
        [Value(1.0), Value(2.0)]
        >>> neuron.bias
        Value(3.0)
        >>> neuron.output
        Value(0.0)
    """
    def __init__(self, weights: list[Value], bias: Value):
        self.weights = weights
        self.bias = bias
        self.output = Value(0.0)

class Layer:
    """
    Represents a layer of neurons.

    Attributes:
        neurons (list[Neuron]): The neurons in the layer.

    Example:
        >>> layer = Layer([Neuron([Value(1.0), Value(2.0)], Value(3.0)), Neuron([Value(4.0), Value(5.0)], Value(6.0))])
        >>> layer.neurons
        [Neuron([Value(1.0), Value(2.0)], Value(3.0)), Neuron([Value(4.0), Value(5.0)], Value(6.0))]
    """
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

class MLP:
    """
    Represents a multi-layer perceptron neural network.

    Attributes:
        layers (list[Layer]): The layers in the network.

    Example:
        >>> mlp = MLP([Layer([Neuron([Value(1.0), Value(2.0)], Value(3.0)), Neuron([Value(4.0), Value(5.0)], Value(6.0))]), Layer([Neuron([Value(7.0), Value(8.0)], Value(9.0))])])
        >>> mlp.layers
        [Layer([Neuron([Value(1.0), Value(2.0)], Value(3.0)), Neuron([Value(4.0), Value(5.0)], Value(6.0))]), Layer([Neuron([Value(7.0), Value(8.0)], Value(9.0))])]
    """
    def __init__(self, layers: list[Layer]):
        self.layers = layers
