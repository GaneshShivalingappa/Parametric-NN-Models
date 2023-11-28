import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from rod.data import create_training_dataset_1D, collate_training_data_1D
from PiNN.network import FFNN
from torch.utils.data import DataLoader 
import statistics
from numpy import linalg
from torch import Tensor
import time
from FEM_test.bc_FEM import NN_FEM
torch.manual_seed(1)


def normalize_input(x,E):
    x_min = min(x)
    X_max = max(x)
    x_nor = (x - x_min)/(X_max-x_min)
    print("X_nor:",x_nor)
    E_min =  min(E)
    E_max = max(E)
    if E_min==E_max:
        E_nor = (E)/torch.mean(E)
    else:
        E_nor = (E - E_min)/(E_max-E_min)
    print("E_nor:",E_nor)
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

    print(u_data)

    u_data = torch.cat(u_data, dim=0)
    u_data = u_data.reshape(-1,1)
    print(u_data)
    

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
            #u = G_u + D_u * self.Nnet(inputs)[:,0]
            u = G_u + D_u *self.Nnet(inputs).reshape(-1)
            u = u * u_data.max()
            #print(u)
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
        #x_e = normalize_input(x,x_e)
        print('input normalized',torch.concat((x, x_e), dim=1))
        u = ansatz(torch.concat((x, x_e), dim=1))
        #u = ansatz(x_e)
        print(u)
        loss = loss_metric(u,u_data)
        print(loss)
        return loss

    #Training

    loss_hist_fem = []
    loss_hist_stress_bc = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

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
    for j in range(10):
        t = FEM_data(no_element=i-1, num_samples_train=64)
        train_time.append(t)
    mean.append(statistics.mean(train_time))
    std.append(statistics.stdev(train_time))

print(mean)
print(std)

quit()

# def prediction_input_normalized(x_cord,E_pred):
#     if (max_youngs_modulus==min_youngs_modulus):
#         E_nor_vec = torch.ones(num_points_pde,1)
#     else :  
#         E_nor = (E_pred-min_youngs_modulus)/(max_youngs_modulus-min_youngs_modulus)
#         E_nor_vec = torch.ones(num_points_pde*2,1)*E_nor
#     input_nor = torch.concat((x_cord, E_nor_vec), dim=1)
#     return input_nor

# # x = Variable(torch.from_numpy(np.linspace(0, 1, num_points_pde).reshape(-1, 1)).float(), requires_grad=True)
# # #inputs1 = {'coords': x, 'youngs_modulus': E1 }
# # E= torch.ones(num_points_pde,1)*youngs_modulus
# # inputs1=prediction_input_normalized(x,youngs_modulus)
# # #inputs1 = torch.concat((x, E), dim=1)
# # print(inputs1)
# # u = ansatz(inputs1)
# # x = torch.linspace(0, 1, num_points_pde)
# # print('predicted1:', u)


# def true_displacement(x, volume_force, force, area, length, youngs):
#     """Generate displacement data."""
#     b = volume_force
#     sig = force / area
#     L = length
#     E = youngs
#     disp = (-b * x**2 + 2 * sig * x + 2 * b * L * x) / (2 * E)
#     return disp

# u_pred = []
# u_real = []
# x = Variable(torch.from_numpy(np.linspace(0, 1, num_points_pde*2 ).reshape(-1, 1)).float(), requires_grad=True)
# E_pred = np.linspace(max_youngs_modulus,min_youngs_modulus,num_samples_train*2)
# print(E_pred)

# for i in E_pred:
#     input_femnn = prediction_input_normalized(x_cord=x,E_pred=i)
#     dis_true = true_displacement(x, Vol_force, force, area, length, i)
#     u_femnn = ansatz(input_femnn)
#     u_fe = u_femnn.detach().numpy()
#     u_r  = dis_true.detach().numpy()
#     u_pred.append(u_fe.reshape(-1,1))
#     u_real.append(u_r.reshape(-1,1))


# u_pred_con = np.concatenate(u_pred)
# u_real_con = np.concatenate(u_real)
# uu = u_pred_con.reshape(num_points_pde*2,num_samples_train*2)
# print(uu)

# u_relative_error = []

# for index, (dis_real, dis_pred) in enumerate(zip(u_real_con, u_pred_con)):

#     if dis_real == 0.0:
#         u_relative_error.append(0.0)
#     else:
#         u_relative_error.append((np.abs(dis_real-dis_pred)*100/dis_real)[0])

# # u_relative_error = np.concatenate(u_relative_error)
# # # print(u_relative)
# u_abs = np.abs(u_real_con - u_pred_con)
# u_abs = np.array(u_abs)
# u_abs = u_abs.reshape(num_points_pde*2,num_samples_train*2)
# u_relative_error = np.array(u_relative_error)
# uu = u_relative_error.reshape(num_points_pde*2,num_samples_train*2)
# x = np.linspace(0,1,num_points_pde*2)
# print(uu)

# plt.figure(1)
# plt.imshow(uu,extent=[x.min(), x.max(), E_pred.min(), E_pred.max()],aspect='auto', interpolation='bilinear',cmap='RdBu',vmin=0.0,vmax=2.0)
# plt.colorbar( label='displacement (u) relative error [%]')
# plt.title("FEM surrogate model")
# plt.xlabel('x-cord [m]')
# plt.ylabel("young's modulus [GPa]")
# plt.savefig('relative_error_FEM_surrogate.png')
# # plt.figure(2)
# # plt.imshow(u_abs,extent=[x.min(), x.max(), E_pred.min(), E_pred.max()],aspect='auto', interpolation='bilinear',cmap='RdBu')
# # plt.colorbar(label='displacement (u) absolute error')
# # plt.title("Physics-Informed Neural Networks")
# # plt.xlabel('x-cord')
# # plt.ylabel("young's modulus (E)")
# # plt.savefig('absolut_error_pinn.png')
# #print(np.reshape(concat, (5,5)))
# quit()
