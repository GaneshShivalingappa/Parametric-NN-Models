from PiNN.network import FFNN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

youngs_modulus = 210.0
force = 100.0
area = 20.0
sig_applied = force/area
Vol_force = 0.0
length = 1.0

class Ansatz(nn.Module):

    def __init__(self, Nnet):
        super().__init__()
        self.Nnet = Nnet

    def forward(self, input):

        G_u = 0.0
        D_u = 0.0 - input[:,0]
        u = G_u + D_u * self.Nnet(input)[:,0]

        return u
    
net = FFNN([1,16,16,1])

anz = Ansatz(net)

optimizer = torch.optim.Adam(anz.parameters(), lr=0.01)

def loss(x):
    
    out = anz(x)
    
    u = out

    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]
    
    residual_balance = youngs_modulus * u_xx 
    residual_bc = sig_applied - youngs_modulus * u_x[-1]

    return {"residual_balance":residual_balance,"residual_bc":residual_bc}

iterations = 500
#x = torch.linspace(0.0, length, 100, requires_grad=True)
x = Variable(torch.from_numpy(np.linspace(0, 1, 100).reshape(-1, 1)).float(), requires_grad=True)
losses = []



for epoch in range(iterations):
    optimizer.zero_grad()
    residual = loss(x)
    r_bal = residual["residual_balance"]
    r_bc = residual["residual_bc"]
    loss_glo = torch.mean(torch.square(r_bal)) + torch.mean(torch.square(r_bc))
    losses.append(loss_glo.item())
    loss_glo.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        print(epoch,"Traning Loss:",loss_glo.data)

u = anz(x)
print(u)

def true_stress(x, volume_force, force, area, length):
    """Generate stress data"""
    b = volume_force
    sig = force / area
    L = length

    stress = -b * x + sig + b * L

    return stress

def true_displacement(x, volume_force, force, area, length, youngs):
    """Generate displacement data."""
    b = volume_force
    sig = force / area
    L = length
    E = youngs

    disp = (-b * x**2 + 2 * sig * x + 2 * b * L * x) / (2 * E)

    return disp

x = np.linspace(0,1,100)
dis_true = true_displacement(x, Vol_force, force, area, length, youngs_modulus)
sig_true = true_stress(x, Vol_force, force, area, length)

ud = u.detach().numpy()
strain = np.gradient(ud, x)
stress = youngs_modulus*strain

plt.figure(0)
plt.semilogy(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.figure(1)
plt.plot(x,ud)
plt.plot(x,dis_true)
plt.title('displacement')
plt.savefig('disp.png')
plt.figure(2)
plt.plot(x,stress)
plt.plot(x,sig_true)
plt.title('Stress')
plt.savefig('stress.png')