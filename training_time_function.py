import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from nn_model.data import create_training_dataset_1D, collate_training_data_1D
from nn_model.network import FFNN
from torch.utils.data import DataLoader 
import statistics
from torch import Tensor
import time
from FEM.fem import NN_FEM, FEM
from nn_model.normalized_hbc_ansatz_1d import create_normalized_hbc_ansatz_1D
from nn_model.momentum import momentum_equation_func,traction_func
from time import perf_counter

torch.manual_seed(1)

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

    u_data = torch.cat(u_data, dim=0)
    u_data = u_data.reshape(-1,1)
    

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
    
    def normalize_input_FEM_Data(x,E):
        x_min = min(x)
        X_max = max(x)
        x_nor = (x - x_min)/(X_max-x_min)
        E_min =  min(E)
        E_max = max(E)
        if E_min==E_max:
            E_nor = (E)/torch.mean(E)
        else:
            E_nor = (E - E_min)/(E_max-E_min)
        return x_nor,E_nor

    def loss_fun(PDE_data):
        x = PDE_data.x_coor
        x_e = PDE_data.x_E
        x,x_e = normalize_input_FEM_Data(x,x_e)
        u = ansatz(torch.concat((x, x_e), dim=1))
        loss = loss_metric(u,u_data)
        return loss

    #Training

    loss_hist_fem = []

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_fun(batch_pde)
        loss.backward()
        return loss.item()

    print("Start training ...")

    st = perf_counter()

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
            print(epoch,"Traning Loss FEM-Data-NN:",loss.detach().item())

    et = perf_counter()
    return et - st, loss_hist_fem

def FEM_NN(no_element,num_samples_train):
    no_element = no_element
    num_samples_train = num_samples_train
    youngs_modulus = 180.0 
    E1 = 180.0 
    E2 = 240.0 
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
    displacement_left = 0.0
    min_displacement = displacement_left
    max_coordinate = length

    max_displacement = calculate_displacements_solution_1D(
        coordinates=max_coordinate,
        length=length,
        youngs_modulus=min_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )

    loss_metric = torch.nn.MSELoss()
    n_node = no_element+1 

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
            u = G_u + D_u * self.Nnet(inputs).reshape(-1)
            return u * max_displacement

    #model
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
        line_search_fn="strong_wolfe",
    )

    x_cor = Variable(torch.from_numpy(np.linspace(0, 1, n_node).reshape(-1, 1)).float(), requires_grad=True)
    K=[]
    F=[]

    E = np.linspace(min_youngs_modulus,max_youngs_modulus,num_samples_train)

    for e in E:
        K_np, F_np = FEM(length=length,no_element=no_element,E=e,T=sig_applied)
        K_np = torch.from_numpy(K_np).type(torch.float)
        F_np = torch.from_numpy(F_np).type(torch.float)
        K.append(K_np)
        F.append(F_np)

    K_main = torch.zeros((n_node*num_samples_train, n_node*num_samples_train), dtype=torch.float32)
    current_row = 0
    current_col = 0
    for tensor in K:
        row_size, col_size = tensor.shape
        K_main[current_row:current_row + row_size, current_col:current_col + col_size] = tensor
        current_row += row_size
        current_col += col_size

    #print(K_main)
    F = torch.cat(F, dim=0)

    def normalize_input(x,E):
        x_min = min(x)
        X_max = max(x)
        x_nor = (x - x_min)/(X_max-x_min)
        E_min =  min(E)
        E_max = max(E)
        if E_min==E_max:
            E_nor = (E)/torch.mean(E)
        else:
            E_nor = (E - E_min)/(E_max-E_min)
        return x_nor,E_nor

    def loss_fun(ansatz,PDE_data):
        x = PDE_data.x_coor
        x_e = PDE_data.x_E
        x,x_e = normalize_input(x,x_e)
        u = ansatz(torch.concat((x, x_e), dim=1))
        A = torch.matmul(K_main,u)
        loss_fem = loss_metric(A,F)
        loss = loss_fem 
        return loss

    #Training

    loss_hist_fem_NN = []

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_fun(ansatz,batch_pde)
        loss.backward()
        return loss.item()

    print("Start training ...")

    st = perf_counter()

    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_fem_batches = []
        loss = []
        for batch_pde, batch_stress_bc in train_batches:
            ansatz.train()
            optimizer.zero_grad()
            loss = loss_fun(ansatz, batch_pde)
            loss.backward()
            optimizer.step(loss_func_closure)
            loss_hist_fem_batches.append(loss.detach().item())

        mean_loss_fem = statistics.mean(loss_hist_fem_batches)
        loss_hist_fem_NN.append(mean_loss_fem)

        with torch.autograd.no_grad():
            print(epoch,"Traning Loss FEM-NN:",loss.detach().item())

    et = perf_counter()

    return et - st, loss_hist_fem_NN

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
    total_loss = []
    def loss_func_closure() -> float:
            optimizer.zero_grad()
            loss_pde, loss_stress_bc = loss_fun(ansatz, batch_pde, batch_stress_bc)
            loss = loss_pde + loss_stress_bc
            loss.backward()
            return loss.item()

    print("Start training ...")
    st = perf_counter()

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
        mean_total_loss = mean_loss_pde + mean_loss_stress_bc
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_stress_bc.append(mean_loss_stress_bc)
        total_loss.append(mean_total_loss)

        with torch.autograd.no_grad():

            print(epoch,"Total Traning Loss:", mean_total_loss)
            #print(epoch,"Traning Loss stress:",mean_loss_stress_bc)

    et = perf_counter()
    return et - st, total_loss
