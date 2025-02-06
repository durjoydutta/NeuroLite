import numpy as np
from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):  # nin : no. of inputs
        self.w = [
            Value(np.random.uniform(-1, 1)) for _ in range(nin)
        ]  # n. no of weights
        self.b = Value(np.random.uniform(-1, 1))  # n. no of inputs

    def __call__(self, x):
        act = (
            sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        )  # raw activation z = w * x + b
        out = act.tanh()  # passing it through acitvation func f(z)/ here tanh(z)
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(
        self, nin, nout
    ):  # builds a layer consisting of 'nout' neurons each with 'nin' inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):  # calls neuron func to get the out expression of each
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(
        self, nin, nouts
    ):  # builds 'len(nouts)' no. of layers with 'nout[i]' no. of neurons at each level and initial 'nin' leaf level inputs
        sz = [
            nin
        ] + nouts  # creating a new list by appending nin alongside nouts, so that we can switch bw inputs and outputs for each layer. eg. in layer1 in: nin / out: nouts[0]
        # in layer2 in: nouts[0] / out: nouts[1] and so on
        self.layers = [
            Layer(sz[i], sz[i + 1]) for i in range(len(nouts))
        ]  # also could have written len(sz) - 1

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
