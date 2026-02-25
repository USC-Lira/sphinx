import torch as th
from torch import nn

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int] = [],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> nn.Module:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output (last layer, for instance, the number of actions)
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return: The list of layers of the neural network
    """

    modules = []
    if len(net_arch) > 0:
        modules.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))
        modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))

    if squash_output:
        modules.append(nn.Tanh())
        
    return nn.Sequential(*modules)