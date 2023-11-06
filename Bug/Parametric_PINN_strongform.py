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

seed = 1
torch.manual_seed(seed)
#n_points = 50
youngs_modulus = 180.0
force = 100.0
area = 20.0
traction = sig_applied = force/area
Vol_force = 0.0
length = 1.0
num_samples_train = 128
num_points_pde = 128
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

train_batches = iter(train_dataloader)


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
        #print(loss_pde,loss_stress_bc )

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

et = time.time()
print('Execution time:', et - st, 'seconds')

x = Variable(torch.from_numpy(np.linspace(0, 1, num_points_pde).reshape(-1, 1)).float(), requires_grad=True)
E1 = torch.ones(num_points_pde,1)*youngs_modulus
#inputs1 = {'coords': x, 'youngs_modulus': E1 }
inputs1=torch.concat((x, E1), dim=1)
print(inputs1)
u = ansatz(inputs1)
print('predicted1:',u)


def prediction_input_normalized(x_cord,E_pred):

    E_nor_vec = torch.ones(num_points_pde*2,1)*E_pred
    input_nor = torch.concat((x_cord, E_nor_vec), dim=1)
    return input_nor

def true_displacement(x, volume_force, force, area, length, youngs):
    """Generate displacement data."""
    b = volume_force
    sig = force / area
    L = length
    E = youngs
    disp = (-b * x**2 + 2 * sig * x + 2 * b * L * x) / (2 * E)
    return disp



u_pred = []
u_real = []
x = Variable(torch.from_numpy(np.linspace(0, 1, num_points_pde*2 ).reshape(-1, 1)).float(), requires_grad=True)
E_pred = np.linspace(max_youngs_modulus,min_youngs_modulus,num_samples_train*2)
print(E_pred)

for i in E_pred:
    input_femnn = prediction_input_normalized(x_cord=x,E_pred=i)
    dis_true = true_displacement(x, Vol_force, force, area, length, i)
    u_femnn = ansatz(input_femnn)
    u_fe = u_femnn.detach().numpy()
    u_r  = dis_true.detach().numpy()
    u_pred.append(u_fe.reshape(-1,1))
    u_real.append(u_r.reshape(-1,1))


u_pred_con = np.concatenate(u_pred)
u_real_con = np.concatenate(u_real)
uu = u_pred_con.reshape(num_points_pde*2,num_samples_train*2)
print(uu)

u_relative_error = []

for index, (dis_real, dis_pred) in enumerate(zip(u_real_con, u_pred_con)):

    if dis_real == 0.0:
        u_relative_error.append(0.0)
    else:
        u_relative_error.append((np.abs(dis_real-dis_pred)*100/dis_real)[0])

# u_relative_error = np.concatenate(u_relative_error)
# # print(u_relative)
u_abs = np.abs(u_real_con - u_pred_con)
u_abs = np.array(u_abs)
u_abs = u_abs.reshape(num_points_pde*2,num_samples_train*2)
u_relative_error = np.array(u_relative_error)
uu = u_relative_error.reshape(num_points_pde*2,num_samples_train*2)
x = np.linspace(0,1,num_points_pde*2)
print(uu)

plt.figure(0)
plt.semilogy(loss_hist_pde)
plt.title(f'Training Loss (n = {num_points_pde})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f'PINNs/loss_PINN_{num_epochs}_{seed}.png')

plt.figure(1)
plt.imshow(
    uu,extent=[x.min(), x.max(), E_pred.min(), E_pred.max()],
    aspect='auto', 
    interpolation='bilinear',
    cmap='RdBu',
    vmin=0.0,
    vmax=0.042,
    )
plt.colorbar(label='displacement (u) relative error [%]')
#plt.title("Physics-Informed Neural Networks")
plt.xlabel('x [m]')
plt.ylabel("young's modulus [GPa]")
plt.savefig(f'PINNs/relative_error_pinn_{num_epochs}_{seed}.png')
# plt.figure(2)
# plt.imshow(u_abs,extent=[x.min(), x.max(), E_pred.min(), E_pred.max()],aspect='auto', interpolation='bilinear',cmap='RdBu')
# plt.colorbar(label='displacement (u) absolute error')
# plt.title("Physics-Informed Neural Networks")
# plt.xlabel('x-cord')
# plt.ylabel("young's modulus (E)")
# plt.savefig('absolut_error_pinn.png')
# #print(np.reshape(concat, (5,5)))
quit()

x = np.linspace(0,1,num_points_pde)
dis_true = true_displacement(x, Vol_force, force, area, length, youngs_modulus)
print(dis_true)
ud = u.detach().numpy()

plt.figure(0)
plt.semilogy(loss_hist_pde)
plt.title(f'Training Loss (n = {num_points_pde}, E_pred = {youngs_modulus})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.figure(1)
plt.plot(x,ud,label='u_pinn',marker='o')
plt.plot(x,dis_true,label='u_Analytical')
plt.title(f'displacement (n = {num_points_pde}, E_pred = {youngs_modulus})')
plt.xlabel('Length')
plt.ylabel('displacement')
plt.legend()
plt.savefig('disp.png')
