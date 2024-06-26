import matplotlib.pyplot as plt
from training_time_function import FEM_data, FEM_NN, PINN
import statistics
import numpy as np

sample = [10,20,30,40,50,60,70,80,90,100]

mean_FEM_NN = []
std_FEM_NN = []
mean_FEM_Data = []
std_FEM_Data = []
mean_PINN = []
std_PINN = []


for i in sample:
    train_time_FEM_Data = []
    train_time_FEM_NN = []
    train_time_PINN = []

    num_iter = 25

    for k in range(num_iter):
        t_FEM_Data, loss_fem_data = FEM_data(no_element=63, num_samples_train=i)
        train_time_FEM_Data.append(t_FEM_Data)

    for l in range(num_iter):
        t_FEM_NN, loss_fem_nn = FEM_NN(no_element=63, num_samples_train=i)
        train_time_FEM_NN.append(t_FEM_NN)
    
    for m in range(num_iter):
        t_PINN, loss_pinn = PINN(no_element=64, num_samples_train=i)
        train_time_PINN.append(t_PINN)

    mean_FEM_Data.append(statistics.mean(train_time_FEM_Data))
    std_FEM_Data.append((statistics.stdev(train_time_FEM_Data))/(np.sqrt(num_iter)))
    mean_FEM_NN.append(statistics.mean(train_time_FEM_NN))
    std_FEM_NN.append((statistics.stdev(train_time_FEM_NN))/(np.sqrt(num_iter)))
    mean_PINN.append(statistics.mean(train_time_PINN))
    std_PINN.append((statistics.stdev(train_time_PINN))/(np.sqrt(num_iter)))

plt.figure(1)
plt.errorbar(sample, mean_PINN, std_PINN, label='PINN', fmt='-o',capsize=2, linewidth=1)
plt.errorbar(sample, mean_FEM_NN, std_FEM_NN, label='FEM-NN', fmt='-o',capsize=2, linewidth=1)
plt.errorbar(sample,mean_FEM_Data,std_FEM_Data,label='FEM-Data-NN', fmt='-o',capsize=2, linewidth=1)
plt.xlabel('Nodes')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,25))
plt.savefig('no_of_sample_time.png')