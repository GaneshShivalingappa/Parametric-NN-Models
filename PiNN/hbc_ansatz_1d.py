from torch import Tensor
from torch.nn import Module
import torch
import torch.nn as nn


class NormalizedHBCAnsatz1D(nn.Module):
    def __init__(
        self,
        displacement_left: Tensor,
        network: Module,
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._network = network

    '''def _boundary_data_func(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer(self._displacement_left)

    def _distance_func(self, x_coor: Tensor) -> Tensor:
        # It is assumed that the HBC is at x_coor=0.
        return self._hbc_ansatz_coordinate_normalizer(x_coor)'''

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return torch.unsqueeze(x[0], 0)
        return torch.unsqueeze(x[:, 0], 1)

    def forward(self, x: Tensor) -> Tensor:
        x_coor = self._extract_coordinates(x)
        D_u = 0.0 - x_coor
        norm_y = self._displacement_left  + (D_u * self._network(x))
        return norm_y

def create_normalized_hbc_ansatz_1D(
    displacement_left: Tensor,
    network: Module,
) :
    return NormalizedHBCAnsatz1D(
        displacement_left=displacement_left,
        network=network
    )