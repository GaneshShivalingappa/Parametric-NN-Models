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
        t_FEM_Data = FEM_data(no_element=i-1, num_samples_train=64)
        train_time_FEM_Data.append(t_FEM_Data)

    for l in range(num_iter):
        t_FEM_NN = FEM_NN(no_element=i-1, num_samples_train=64)
        train_time_FEM_NN.append(t_FEM_NN)
    
    for m in range(num_iter):
        t_PINN = PINN(no_element=i, num_samples_train=64)
        train_time_PINN.append(t_PINN)

    mean_FEM_Data.append(statistics.mean(train_time_FEM_Data))
    std_FEM_Data.append((statistics.stdev(train_time_FEM_Data))/(np.sqrt(num_iter)))
    mean_FEM_NN.append(statistics.mean(train_time_FEM_NN))
    std_FEM_NN.append((statistics.stdev(train_time_FEM_NN))/(np.sqrt(num_iter)))
    mean_PINN.append(statistics.mean(train_time_PINN))
    std_PINN.append((statistics.stdev(train_time_PINN))/(np.sqrt(num_iter)))

plt.figure(1)
plt.errorbar(sample, mean_PINN, std_PINN, label='FEM-NN', fmt='-o',capsize=2, linewidth=1)
plt.errorbar(sample, mean_FEM_NN, std_FEM_NN, label='PINN', fmt='-o',capsize=2, linewidth=1)
plt.errorbar(sample,mean_FEM_Data,std_FEM_Data,label='FEM-Data', fmt='-o',capsize=2, linewidth=1)
plt.xlabel('Nodes')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,25))
plt.savefig('No of node time.png')