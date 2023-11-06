from typing import Callable, TypeAlias

import torch
from torch import vmap
from torch.func import grad

from torch.nn import Module
from torch import Tensor

TModule: TypeAlias = Callable[[Tensor, Tensor], Tensor]


def FEM_func(ansatz: Module, x_coors: Tensor, x_params: Tensor, K:Tensor):
    #_ansatz = _transform_ansatz(ansatz)
    #vmap_func = lambda _x_coor, _x_param: _FEM_func(_ansatz, _x_coor, _x_param,K)
    #return vmap(vmap_func)(x_coors, x_params)
    #disp = _displacement_func(_ansatz,x_coors,x_params)
    #print("input", torch.concat((x_coors, x_params), dim=1))
    disp = ansatz(torch.concat((x_coors, x_params), dim=1))
    print(disp)
    # quit()
    #print(torch.matmul(K,disp))
    return torch.matmul(K,disp)

'''def _FEM_func(ansatz, x_coor, x_param,K):
    displacement_func = _displacement_func(ansatz, x_coor, x_param)[0]
    print('disp:',displacement_func.data)
    quit()
    return torch.matmul(K,displacement_func)'''

def momentum_equation_func(
    ansatz: Module, x_coors: Tensor, x_params: Tensor) -> Tensor:
    _ansatz = _transform_ansatz(ansatz)
    vmap_func = lambda _x_coor, _x_param: _momentum_equation_func(
        _ansatz, _x_coor, _x_param,
    )
    return vmap(vmap_func)(x_coors, x_params)


def traction_func(ansatz: Module, x_coors: Tensor, x_params: Tensor) -> Tensor:
    _ansatz = _transform_ansatz(ansatz)
    vmap_func = lambda _x_coor, _x_param: _traction_func(_ansatz, _x_coor, _x_param)
    return vmap(vmap_func)(x_coors, x_params)


def _momentum_equation_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    x_E = x_param
    return x_E * _u_xx_func(ansatz, x_coor, x_param)


def _traction_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    x_E = x_param
    return x_E * _u_x_func(ansatz, x_coor, x_param)


def _u_x_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    displacement_func = lambda _x_coor: _displacement_func(ansatz, _x_coor, x_param)[0]
    return grad(displacement_func, argnums=0)(x_coor)


def _u_xx_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    u_x_func = lambda _x_coor: _u_x_func(ansatz, _x_coor, x_param)[0]
    return grad(u_x_func, argnums=0)(x_coor)


def _displacement_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    return ansatz(x_coor, x_param)


def _transform_ansatz(ansatz: Module) -> TModule:
    return lambda x_coor, x_param: ansatz(torch.concat((x_coor, x_param), dim=0))