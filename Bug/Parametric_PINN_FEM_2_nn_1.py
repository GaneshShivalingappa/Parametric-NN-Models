import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from rod.data import create_training_dataset_1D, collate_training_data_1D
from PiNN.network import FFNN
from FEM_test.bc_FEM import FEM
from torch.utils.data import DataLoader 
import statistics
from numpy import linalg

torch.manual_seed(1)
no_element = 99
youngs_modulus = 185.0
force = 100.0
area = 20.0
traction = sig_applied = force/area
Vol_force = 0.0
length = 10.0
num_samples_train = 1
num_points_pde = no_element+1
batch_size_train = num_samples_train
num_epochs = 10
volume_force = 0.0
min_youngs_modulus = 185.0
max_youngs_modulus =183.0
displacement_left = 0.0
loss_metric = torch.nn.MSELoss()
#K_np, F_np = FEM(length=length,no_element=no_element,E=youngs_modulus,T=sig_applied)
n_node = no_element+1

'''def normalize_input(x,E):
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
    return x_nor,E_nor'''
    
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
        #print(u)
        return u

#model
layer_sizes = [2,8,1]

F_nn= FFNN(layer_sizes=layer_sizes)
ansatz = Ansatz(F_nn)

optimizer = torch.optim.LBFGS(ansatz.parameters())
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

#print(K)
K_main = torch.zeros((n_node*num_samples_train, n_node*num_samples_train), dtype=torch.float32)
#print(K_main)
current_row = 0
current_col = 0
for tensor in K:
    row_size, col_size = tensor.shape
    K_main[current_row:current_row + row_size, current_col:current_col + col_size] = tensor
    current_row += row_size
    current_col += col_size

print(K_main)
#K = torch.cat(K, dim=1)
#print(K)
F = torch.cat(F, dim=0)
print(F)
u = linalg.solve(K_main,F)
print("np sol:\n", u)

def loss_fun(ansatz,PDE_data):
    x = PDE_data.x_coor
    #E = torch.ones(num_points_pde,1)*youngs_modulus
    x_e = PDE_data.x_E
    # e = x_e[0,0]
    # K_np, F_np = FEM(length=length,no_element=no_element,E=e.item(),T=sig_applied)
    # K = torch.from_numpy(K_np).type(torch.float)
    # F = torch.from_numpy(F_np).type(torch.float)
    #x,E = normalize_input(x,E)
    print(torch.concat((x, x_e), dim=1))
    u = ansatz(torch.concat((x, x_e), dim=1))
    print(u)
    A = torch.matmul(K_main,u)
    #print(B)
    #print(A)
    loss_fem = loss_metric(A,F)
    return loss_fem

#Training

loss_hist_fem = []
loss_hist_stress_bc = []
valid_hist_mae = []
valid_hist_rl2 = []
valid_epochs = []

def loss_func_closure() -> float:
    optimizer.zero_grad()
    loss = loss_fun(ansatz,batch_pde)
    loss.backward()
    return loss.item()

print("Start training ...")

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
    loss_hist_fem.append(mean_loss_fem)

    with torch.autograd.no_grad():
        print(epoch,"Traning Loss pde:",loss.detach().item())


x = Variable(torch.from_numpy(np.linspace(0, 1, n_node ).reshape(-1, 1)).float(), requires_grad=True)
E1 = torch.ones(num_points_pde,1)*youngs_modulus
#inputs1 = {'coords': x, 'youngs_modulus': E1 }
inputs1=torch.concat((x, E1), dim=1)
#print(inputs1)
u = ansatz(inputs1)
print('predicted1:',u)
#print('K*u:',torch.matmul(K,u))


def true_displacement(x, volume_force, force, area, length, youngs):
    """Generate displacement data."""
    b = volume_force
    sig = force / area
    L = length
    E = youngs
    disp = (-b * x**2 + 2 * sig * x + 2 * b * L * x) / (2 * E)
    return disp

x = np.linspace(0,1,num_points_pde)
dis_true = true_displacement(x, Vol_force, force, area, length, youngs_modulus)
print("True disp:", dis_true)
ud = u.detach().numpy()

plt.figure(0)
plt.semilogy(loss_hist_fem)
plt.title(f'Training Loss (n = {n_node})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_FEM_2_nn_1.png')
plt.figure(1)
plt.plot(x,ud,label='u_pinn_FEM',marker='o')
plt.plot(x,dis_true,label='u_Analytical')
plt.title(f'displacement (n = {n_node})')
plt.xlabel('Length')
plt.ylabel('displacement')
plt.legend()
plt.savefig('disp_FEM_2_nn_1.png')