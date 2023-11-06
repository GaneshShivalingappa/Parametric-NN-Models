import torch
import torch.nn as nn
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
#n_points = 50
youngs_modulus = 231.0
force = 100.0
area = 20.0
sig_applied = force/area
Vol_force = 0.0
length = 1.0
num_samples_train = 5
num_points_pde = 10
batch_size_train = num_samples_train
num_epochs = 10
traction = 5.0
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
ansatz = create_normalized_hbc_ansatz_1D(
    displacement_left=torch.tensor([displacement_left]),
    network=F_nn,
    min_inputs=min_inputs,
    max_inputs=max_inputs,
    min_outputs=min_output,
    max_outputs=max_output,
)

#optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
optimizer = torch.optim.LBFGS(ansatz.parameters())

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

for epoch in range(num_epochs):
    train_batches = iter(train_dataloader)
    loss_hist_pde_batches = []
    loss_hist_stress_bc_batches = []


    for batch_pde, batch_stress_bc in train_batches:

        ansatz.train()

        loss_pde, loss_stress_bc = loss_fun(ansatz, batch_pde, batch_stress_bc)
        print(loss_pde,loss_stress_bc )

        optimizer.step(loss_func_closure)

        loss_hist_pde_batches.append(loss_pde.detach().item())
        loss_hist_stress_bc_batches.append(loss_stress_bc.detach().item())
    
    mean_loss_pde = statistics.mean(loss_hist_pde_batches)
    mean_loss_stress_bc = statistics.mean(loss_hist_stress_bc_batches)
    loss_hist_pde.append(mean_loss_pde)
    loss_hist_stress_bc.append(mean_loss_stress_bc)

    with torch.autograd.no_grad():

        print(epoch,"Traning Loss pde:",mean_loss_pde)
        print(epoch,"Traning Loss stress:",mean_loss_stress_bc)


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
plt.semilogy(loss_hist_pde)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.figure(1)
plt.plot(x,ud,label='u_pinn',marker='o')
plt.plot(x,dis_true,label='u_Analytical')
plt.title('displacement')
plt.xlabel('Length')
plt.ylabel('displacement')
plt.legend()
plt.savefig('disp.png')