import torch.nn as nn
from torch import Tensor



class HBCAnsatzNormalizer(nn.Module):
    def __init__(self, min_values: Tensor, max_values: Tensor) -> None:
        super().__init__()
        self._value_ranges = max_values - min_values

    def forward(self, x: Tensor) -> Tensor:
        return x / self._value_ranges


class HBCAnsatzRenormalizer(nn.Module):
    def __init__(self, min_values: Tensor, max_values: Tensor) -> None:
        super().__init__()
        self._value_ranges = max_values - min_values

    def forward(self, normalized_x: Tensor) -> Tensor:
        return normalized_x * self._value_ranges
