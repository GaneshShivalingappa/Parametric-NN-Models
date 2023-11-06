import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


class InputNormalizer(nn.Module):
    def __init__(self, min_inputs: Tensor, max_inputs: Tensor) -> None:
        super().__init__()
        self._min_inputs = min_inputs
        self._max_inputs = max_inputs
        self._input_ranges = max_inputs - min_inputs
        self._atol = torch.tensor([1e-7]).to(self._input_ranges.device)

    def forward(self, x: Tensor) -> Tensor:
        return ((((x - self._min_inputs) + self._atol)/ (self._input_ranges + self._atol)))
    

class OutputRenormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._min_outputs = min_outputs
        self._max_outputs = max_outputs
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return (((x + 1) / 2) * self._output_ranges) + self._min_outputs


class NormalizedNetwork(nn.Module):
    def __init__(
        self,
        network: Module,
        input_normalizer: Module,
        output_renormalizer: Module,
    ) -> None:
        super().__init__()
        self._network = network
        self._input_normalizer = input_normalizer
        self._output_renormalizer = output_renormalizer

    def forward(self, x: Tensor) -> Tensor:
        normalized_x = self._input_normalizer(x)
        normalized_y = self._network(normalized_x)
        y = self._output_renormalizer(normalized_y)
        return y


def create_normalized_network(
    network: Module,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedNetwork:
    input_normalizer = InputNormalizer(min_inputs, max_inputs)
    output_renormalizer = OutputRenormalizer(min_outputs, max_outputs)
    return NormalizedNetwork(
        network=network,
        input_normalizer=input_normalizer,
        output_renormalizer=output_renormalizer,
    )

