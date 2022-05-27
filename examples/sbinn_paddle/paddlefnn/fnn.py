# code was heavily based on https://github.com/lululxvi/deepxde
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lululxvi/deepxde#license

import paddle
from .nn import NN

class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes):
        super().__init__()
        self.activation = paddle.nn.Swish()
        self.linears =paddle.nn.LayerList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                paddle.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )
    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.linears[:-1]:
            # temp = linear(x)
            # x = temp*self.activation(temp)
            # temp = linear(x)
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

    def requires_grad_(self,requires_grad=True):
        if requires_grad:
            for params in self.parameters():
                params.trainable = True
        else:
            for params in self.parameters():
                params.trainable = False
