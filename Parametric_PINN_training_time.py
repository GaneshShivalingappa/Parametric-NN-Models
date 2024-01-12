import torch
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from torch.autograd import Variable
from rod.data import create_training_dataset_1D, collate_training_data_1D
from PiNN.network import FFNN
from PiNN.normalized_hbc_ansatz_1d import create_normalized_hbc_ansatz_1D
from rod.momentum import momentum_equation_func,traction_func
from torch.utils.data import DataLoader 
import statistics
from torch import Tensor
import time

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


torch.manual_seed(1)

def PINN(no_element, num_samples_train):

    youngs_modulus = 200.0
    force = 100.0
    area = 20.0
    traction = sig_applied = force/area
    Vol_force = 0.0
    length = 1.0
    num_samples_train = num_samples_train
    num_points_pde = no_element
    batch_size_train = num_samples_train
    num_epochs = 32
    volume_force = 0.0
    min_youngs_modulus = 180.0
    max_youngs_modulus = 240.0
    displacement_left = 0.0
    loss_metric = torch.nn.MSELoss()

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

    #model
    layer_sizes = [2, 8, 1]
    min_coordinate = 0.0
    max_coordinate = length
    min_displacement = displacement_left
    max_displacement = calculate_displacements_solution_1D(
        coordinates=max_coordinate,
        length=length,
        youngs_modulus=min_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    min_inputs = torch.tensor([min_coordinate, min_youngs_modulus])
    max_inputs = torch.tensor([max_coordinate, max_youngs_modulus])
    min_output = torch.tensor([min_displacement])
    max_output = torch.tensor([max_displacement])
    F_nn= FFNN(layer_sizes=layer_sizes)
    ansatz = create_normalized_hbc_ansatz_1D(
        displacement_left=torch.tensor([displacement_left]),
        network=F_nn,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_output,
        max_outputs=max_output,
    )

    optimizer = torch.optim.LBFGS(
                    ansatz.parameters(),lr=1.0,
                    max_iter=20,
                    max_eval=25,
                    tolerance_grad=1e-9,
                    tolerance_change=1e-12,
                    history_size=100,
                    line_search_fn="strong_wolfe"
                    )

    def loss_fun(ansatz ,pde_data,stress_bc_data)-> tuple[Tensor, Tensor]:
        network = ansatz 
        pde_data = pde_data
        stress_bc_data =stress_bc_data

        def loss_func_pde(network, pde_data) -> Tensor:
            x_coor = pde_data.x_coor
            x_E = pde_data.x_E
            y_true = pde_data.y_true
            y = momentum_equation_func(network, x_coor, x_E)
            return loss_metric(y_true, y)


        def loss_func_stress_bc(network, stress_bc_data) -> Tensor:
            x_coor = stress_bc_data.x_coor
            x_E = stress_bc_data.x_E
            y_true = stress_bc_data.y_true
            y = traction_func(network, x_coor, x_E)
            return loss_metric(y_true, y)

        loss_pde = loss_func_pde(network, pde_data)
        loss_stress_bc = loss_func_stress_bc(network, stress_bc_data)
        return loss_pde, loss_stress_bc


    #Training

    loss_hist_pde = []
    loss_hist_stress_bc = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []
    def loss_func_closure() -> float:
            optimizer.zero_grad()
            loss_pde, loss_stress_bc = loss_fun(ansatz, batch_pde, batch_stress_bc)
            loss = loss_pde + loss_stress_bc
            loss.backward()
            return loss.item()

    print("Start training ...")
    st = time.time()

    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_stress_bc_batches = []


        for batch_pde, batch_stress_bc in train_batches:

            ansatz.train()

            loss_pde, loss_stress_bc = loss_fun(ansatz, batch_pde, batch_stress_bc)

            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().item())
            loss_hist_stress_bc_batches.append(loss_stress_bc.detach().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_stress_bc = statistics.mean(loss_hist_stress_bc_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_stress_bc.append(mean_loss_stress_bc)

        with torch.autograd.no_grad():

            print(epoch,"Traning Loss pde:",mean_loss_pde)
            #print(epoch,"Traning Loss stress:",mean_loss_stress_bc)

    et = time.time()
    return et - st

sample = [10,20,30,40,50,60,70,80,90,100]

mean = []
std = []

for i in sample:
    train_time = []
    for j in range(25):
        t = PINN(no_element=64, num_samples_train=i)
        train_time.append(t)
    mean.append(statistics.mean(train_time))
    std.append((statistics.stdev(train_time))/(np.sqrt(25)))

print(mean)
print(std)
print(torch.cuda.is_available())