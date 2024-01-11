import torch
import torch.nn as nn
import numpy as np
from rod.data import create_training_dataset_1D, collate_training_data_1D
from PiNN.network import FFNN
from torch.utils.data import DataLoader 
import statistics
from torch import Tensor
import time
from FEM_test.bc_FEM import NN_FEM
torch.manual_seed(1)

def normalize_input(x,E):
    x_min = min(x)
    X_max = max(x)
    x_nor = (x - x_min)/(X_max-x_min)
    #print("X_nor:",x_nor)
    E_min =  min(E)
    E_max = max(E)
    if E_min==E_max:
        E_nor = (E)/torch.mean(E)
    else:
        E_nor = (E - E_min)/(E_max-E_min)
    #print("E_nor:",E_nor)
    return x_nor,E_nor
    #return E_nor


def calculate_displacements_solution_1D(
    coordinates: Tensor | float,
    length: float,
    youngs_modulus: Tensor | float,
    traction: float,
    volume_force: float,
) -> Tensor | float:
    return (traction / youngs_modulus) * coordinates + (
        volume_force / youngs_modulus
    ) * (length * coordinates - 1 / 2 * coordinates**2)

def FEM_data(no_element, num_samples_train):
    no_element = no_element
    num_samples_train = num_samples_train
    youngs_modulus = 240.0
    force = 100.0
    area = 20.0
    traction = sig_applied = force/area
    Vol_force = 0.0
    length = 1.0
    num_points_pde = no_element+1
    batch_size_train = num_samples_train
    num_epochs = 32
    volume_force = 0.0
    min_youngs_modulus = 180.0
    max_youngs_modulus =240.0

    u_data = []

    u = NN_FEM(length=length,no_element=no_element,E=youngs_modulus,T=sig_applied)
    E = np.linspace(min_youngs_modulus,max_youngs_modulus,num_samples_train)

    for e in E:
        u = NN_FEM(length=length,no_element=no_element,E=e,T=sig_applied)
        u = torch.from_numpy(u).type(torch.float)
        u_data.append(u)

    #print(u_data)

    u_data = torch.cat(u_data, dim=0)
    u_data = u_data.reshape(-1,1)
    #print(u_data)
    

    print("Create training data ...")
    train_dataset = create_training_dataset_1D(length=length,
            traction=traction,
            volume_force=volume_force,
            min_youngs_modulus=min_youngs_modulus,
            max_youngs_modulus=max_youngs_modulus,
            num_points_pde=num_points_pde,
            num_samples=num_samples_train
            )

    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size_train,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_training_data_1D,
        )

    class Ansatz(nn.Module):

        def __init__(self, Nnet):
            super().__init__()
            self.Nnet = Nnet

        def forward(self, inputs):
            x_coor = inputs[:, 0]
            G_u = 0.0
            D_u = 0.0 - x_coor
            u = G_u + D_u *self.Nnet(inputs).reshape(-1)
            u = u * u_data.max()
            return u.reshape(-1,1)

    loss_metric = torch.nn.MSELoss()
    layer_sizes = [2,8,1]
    F_nn= FFNN(layer_sizes=layer_sizes)
    ansatz = Ansatz(F_nn)
    optimizer = torch.optim.LBFGS(params=ansatz.parameters(),
                    lr=1.0,
                    max_iter=20,
                    max_eval=25,
                    tolerance_grad=1e-9,
                    tolerance_change=1e-12,
                    history_size=100,
                    line_search_fn="strong_wolfe")

    def loss_fun(PDE_data):
        x = PDE_data.x_coor
        x_e = PDE_data.x_E
        x,x_e = normalize_input(x,x_e)
        #print('input normalized',torch.concat((x, x_e), dim=1))
        u = ansatz(torch.concat((x, x_e), dim=1))
        #print(u)
        loss = loss_metric(u,u_data)
        #print(loss)
        return loss

    #Training

    loss_hist_fem = []

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_fun(batch_pde)
        loss.backward()
        return loss.item()

    print("Start training ...")

    st = time.time()

    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_fem_batches = []
        loss = []
        for batch_pde, batch_stress_bc in train_batches:
            F_nn.train()
            optimizer.zero_grad()
            loss = loss_fun(batch_pde)
            loss.backward()
            optimizer.step(loss_func_closure)
            loss_hist_fem_batches.append(loss.detach().item())

        mean_loss_fem = statistics.mean(loss_hist_fem_batches)
        loss_hist_fem.append(mean_loss_fem)

        with torch.autograd.no_grad():
            print(epoch,"Traning Loss pde:",loss.detach().item())

    et = time.time()
    return et - st


sample = [10,20,30,40,50,60,70,80,90,100]

mean = []
std = []

for i in sample:
    train_time = []
    for j in range(100):
        t = FEM_data(no_element=63, num_samples_train=i)
        train_time.append(t)
    mean.append(statistics.mean(train_time))
    std.append((statistics.stdev(train_time))/(np.sqrt(100)))

print("Mean:", mean)
print("std:",std)
print(torch.cuda.is_available())