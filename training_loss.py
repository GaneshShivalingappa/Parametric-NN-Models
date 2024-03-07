import matplotlib.pyplot as plt
from training_time_function import FEM_data, FEM_NN, PINN

t_FEM_Data, loss_fem_data = FEM_data(no_element=127, num_samples_train=128)


t_FEM_NN, loss_fem_nn = FEM_NN(no_element=127, num_samples_train=128)
   
t_PINN, loss_pinn = PINN(no_element=128, num_samples_train=128)

print(loss_fem_data)
print(loss_fem_nn) 
print(loss_pinn)

plt.figure(0)
plt.semilogy(loss_fem_nn, label="FEM-NN")
plt.semilogy(loss_pinn, label = "PINN")
plt.semilogy(loss_fem_data, label = "FEM-Data-NN")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'Training_loss_semilog.png')

plt.figure(1)
plt.plot(loss_fem_nn, label="FEM-NN")
plt.plot(loss_pinn, label = "PINN")
plt.plot(loss_fem_data, label = "FEM-Data-NN")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'Training_loss_plot.png')