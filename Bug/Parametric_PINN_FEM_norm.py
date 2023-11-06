import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from torch.autograd import Variable
from rod.data import create_training_dataset_1D, collate_training_data_1D
from PiNN.network import FFNN
from PiNN.normalized_hbc_ansatz_1d import create_normalized_hbc_FEM_ansatz_1D
from rod.momentum import  FEM_func
from torch.utils.data import DataLoader 
from FEM_test.bc_FEM import FEM
import statistics
from torch.nn import Module
from torch import Tensor

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
youngs_modulus = 190.0
force = 100.0
area = 20.0
sig_applied = traction = force/area
Vol_force = 0.0
length = 1.0
no_element = 4
n_node = no_element+1 
num_points_pde = n_node
num_samples_train = 5
batch_size_train = num_samples_train
num_epochs = 10
volume_force = 0.0
min_youngs_modulus = 180.0
max_youngs_modulus = 240.0
displacement_left = 0.0
loss_metric = torch.nn.MSELoss()


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

print(K_main)

#K_main = torch.cat(K, dim=0)
#print(K)
F = torch.cat(F, dim=0)
print(F)

#F_T = F[no_element::no_element+1]
# print(F_T)
# u = linalg.solve(K_main,F)
# print("np sol:\n", u)

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

'''class Ansatz(nn.Module):

    def __init__(self, Nnet):
        super().__init__()
        self.Nnet = Nnet

    def forward(self, inputs):
        
        x_coor = inputs[0]
        x_coor=torch.unsqueeze(x_coor, 0)
        print(x_coor)
        x_E = inputs[1]
        #x = torch.cat([x_coor, x_E], dim=1)

        G_u = 0.0
        D_u = 0.0 - x_coor[:,0]
        u = G_u + D_u * self.Nnet(inputs)[:,0]
        return u'''

'''class Ansatz(nn.Module):

    def __init__(self, Nnet):
        super().__init__()
        self.Nnet = Nnet

    def forward(self, inputs):
        x_coor = inputs[:, 0]
        G_u = 0.0
        D_u = 0.0 - x_coor
        u = G_u + D_u * self.Nnet(inputs).reshape(-1)
        #print(u)
        return u'''

#model
layer_sizes = [2, 8, 1]
#net = FFNN([2, 32, 1])
#network = Ansatz(net)

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
ansatz = create_normalized_hbc_FEM_ansatz_1D(
    displacement_left=torch.tensor([displacement_left]),
    network=F_nn,
    min_inputs=min_inputs,
    max_inputs=max_inputs,
    min_outputs=min_output,
    max_outputs=max_output,
)

#optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
optimizer = torch.optim.LBFGS(ansatz.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe")

def loss_fun(ansatz ,fem_data)-> tuple[Tensor]:
    network = ansatz 
    fem_data = fem_data

    def loss_func_FEM(network, fem_data) -> Tensor:
        x_coor = fem_data.x_coor
        x_E = fem_data.x_E
        y = FEM_func(network, x_coor, x_E, K_main)
        return loss_metric(F, y)
    
    loss_pde = loss_func_FEM(network, fem_data)
    return loss_pde

#Training

loss_hist_fem = []
loss_hist_stress_bc = []
valid_hist_mae = []
valid_hist_rl2 = []
valid_epochs = []
def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss_fem = loss_fun(ansatz, batch_fem)
        loss = loss_fem 
        loss.backward()
        return loss.item()

print("Start training ...")

for epoch in range(num_epochs):
    train_batches = iter(train_dataloader)
    loss_hist_fem_batches = []
    loss_hist_stress_bc_batches = []

    for batch_fem, batch_stress_bc in train_batches:

        ansatz.train()

        loss_fem = loss_fun(ansatz, batch_fem)
        print(loss_fem)

        optimizer.step(loss_func_closure)

        loss_hist_fem_batches.append(loss_fem.detach().item())
        #loss_hist_stress_bc_batches.append(loss_stress_bc.detach().item())
    
    mean_loss_fem = statistics.mean(loss_hist_fem_batches)
    #mean_loss_stress_bc = statistics.mean(loss_hist_stress_bc_batches)
    loss_hist_fem.append(mean_loss_fem)
    #loss_hist_stress_bc.append(mean_loss_stress_bc)

    with torch.autograd.no_grad():

        print(epoch,"Traning Loss pde:",mean_loss_fem)
        #print(epoch,"Traning Loss stress:",mean_loss_stress_bc)

x = Variable(torch.from_numpy(np.linspace(0, 1, num_points_pde).reshape(-1, 1)).float(), requires_grad=True)
E1 = torch.ones(num_points_pde,1)*youngs_modulus
#inputs1 = {'coords': x, 'youngs_modulus': E1 }
inputs1=torch.concat((x, E1), dim=1)
print(inputs1)
u = ansatz(inputs1)
print('predicted1:',u)

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
print(dis_true)
ud = u.detach().numpy()

plt.figure(0)
plt.semilogy(loss_hist_fem)
plt.title(f'Training Loss (n = {n_node}, E_pred = {youngs_modulus})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_FEM_2_nn_1.png')
plt.figure(1)
plt.plot(x,ud,label='u_pinn_FEM',marker='o')
plt.plot(x,dis_true,label='u_Analytical')
plt.title(f'displacement (n = {n_node}, E_pred = {youngs_modulus})')
plt.xlabel('Length')
plt.ylabel('displacement')
plt.legend()
plt.savefig('disp_FEM_2_nn_1.png')