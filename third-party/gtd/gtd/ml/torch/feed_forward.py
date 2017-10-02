import torch
from torch.nn import Module, Linear


class FeedForwardNetwork(Module):
    """A standard feedforward network, with residual connections for equal-sized layers."""
    def __init__(self, layer_dims):
        """Construct network.
        
        For len(layer_dims) == 3:
        
            y = f(x * W1 + b1) * W2 + b2
        
        x: (batch_size, layer_dims[0])
        W1: (layer_dims[0], layer_dims[1])
        W2: (layer_dims[1], layer_dims[2])
        
        Note that there is no nonlinearity after final linear transform.
        
        Args:
            layer_dims (list[int]):
                layer_dims[0] = input dimension
                layer_dims[-1] = output dimension
        """
        if len(layer_dims) < 3:
            raise ValueError("len(layer_dims) == 2 is just linear, and fewer layers does not make sense.")

        super(FeedForwardNetwork, self).__init__()
        self.nonlinearity = torch.nn.Tanh()  # same for all layers
        self.layers = []
        for i in range(len(layer_dims) - 1):
            # these layers include a bias term
            layer = Linear(layer_dims[i], layer_dims[i + 1])
            # make sure to register sub-module
            self.add_module('linear_{}'.format(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x_prev = x
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.nonlinearity(x)  # apply nonlinearity if it is not the final layer

            if x.size() == x_prev.size():
                x = x + x_prev  # residual connection

        return x