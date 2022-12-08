class Value:
    def __init__(self, data, grad_fn=None):
        if not type(data) == float:
            msg = "data must be a scalar float; tensors and implicit conversion not supported"
            raise ValueError(msg)

        self.data = data
        self.grad_fn = grad_fn
        self.grad = None

    def __add__(self, other):
        grad_fn = AddBackward(operands=(self, other))
        return Value(self.data + other.data, grad_fn=grad_fn)

    def __sub__(self, other):
        return Value(self.data - other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)

    def __truediv__(self, other):
        return Value(self.data / other.data)

    def __pow__(self, power):
        return Value(self.data**power)

    def relu(self):
        if self.data < 0.0:
            return Value(0.0)

        return self

    def backward(self):
        self.grad = 1.0


class AddBackward:
    def __init__(self, operands):
        self.operands = operands

class MulBackward:
    def __init__(self, operands):
        self.operands = operands
