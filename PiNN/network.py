import torch.nn as nn
from torch import Tensor


class LinearHiddenLayer(nn.Module):
    def __init__(self, size_input: int, size_output: int) -> None:
        super().__init__()
        self._fc_layer = nn.Linear(
            in_features=size_input,
            out_features=size_output,
            bias=True,
        )
        nn.init.xavier_uniform_(self._fc_layer.weight)
        nn.init.zeros_(self._fc_layer.bias)
        self._activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self._activation(self._fc_layer(x))


class LinearOutputLayer(nn.Module):
    def __init__(self, size_input: int, size_output: int) -> None:
        super().__init__()
        self._fc_layer = nn.Linear(
            in_features=size_input,
            out_features=size_output,
            bias=True,
        )
        nn.init.xavier_uniform_(self._fc_layer.weight)
        nn.init.zeros_(self._fc_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._fc_layer(x)


class FFNN(nn.Module):
    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__()
        self._layer_sizes = layer_sizes
        layers = self._nest_layers()
        self._output = nn.Sequential(*layers)

    def _nest_layers(self) -> list[nn.Module]:
        layers: list[nn.Module] = [
            LinearHiddenLayer(
                size_input=self._layer_sizes[i - 1],
                size_output=self._layer_sizes[i],
            )
            for i in range(1, len(self._layer_sizes) - 1)
        ]

        layer_out = LinearOutputLayer(
            size_input=self._layer_sizes[-2],
            size_output=self._layer_sizes[-1],
        )
        layers.append(layer_out)
        return layers

    def forward(self, x: Tensor) -> Tensor:
        return self._output(x)
