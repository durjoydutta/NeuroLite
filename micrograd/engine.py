import math


class Value:
    """
    ** Micrograd Engine **
    stores a single scalar value and its gradient (tbd)
    """

    def __init__(
        self, data, _children=(), _op="", label=""
    ) -> None:  # data, children, operations
        self.data = data
        self.grad = 0
        # initially set to zero assuming no affect on output
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data = {self.data})"

    def __str__(self) -> str:
        return f"{self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * (-1.0)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other) ** -1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(math.pow(self.data, other), (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp()")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), _op="tanh")  # tanh object has only single tuple

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        vis = set()
        topo = []

        def build_topo(u):
            vis.add(u)
            for v in u._prev:
                if v not in vis:
                    build_topo(v)
            topo.append(u)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
