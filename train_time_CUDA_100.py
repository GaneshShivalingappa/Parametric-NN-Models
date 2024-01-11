import matplotlib.pyplot as plt

time = [10,20,30,40,50,60,70,80,90,100]

FENN_mean_25_64_node_32 = 
FENN_std_25_64_node_32 = 

FENN_mean_25_64_training_sample_32 = 
FENN_std_25_64_training_sample_32 = 

PINN_mean_25_64_training_sample_32 = 
PINN_std_25_64_training_sample_32 = 

PINN_mean_100_64_node_32 = [5.6987270474433895, 6.204279203414917, 6.3684497284889225, 6.129468684196472, 6.068838346004486, 6.2850825572013855, 6.622608978748321, 6.482682263851165, 7.305266819000244, 7.23656489610672]
PINN_std_100_64_node_32 = [0.18422218304077662, 0.1843984013378021, 0.17541411542569882, 0.16357380562335172, 0.1704275325666258, 0.1764999925951085, 0.16920468993826418, 0.15813881323897674, 0.19741751049845738, 0.18872568019550712]

FEM_data_mean_25_64_nodes_32 = 
FEM_data_std_25_64_nodes_32  = 

FEM_data_mean_25_64_training_sample_32 = 
FEM_data_std_25_64_training_sample_32 = 

plt.figure(1)
plt.errorbar(time,FENN_mean_25_64_training_sample_32,FENN_std_25_64_training_sample_32,label='FEM-NN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,PINN_mean_25_64_training_sample_32,PINN_std_25_64_training_sample_32,label='PINN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FEM_data_mean_25_64_training_sample_32,FEM_data_std_25_64_training_sample_32,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
plt.xlabel('Nodes')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,100))
plt.savefig('No of node time_25.png')


plt.figure(2)
plt.errorbar(time,FENN_mean_25_64_node_32,FENN_std_25_64_node_32,label='FEM-NN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,PINN_mean_25_64_node_32,PINN_std_25_64_node_32,label='PINN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FEM_data_mean_25_64_nodes_32,FEM_data_std_25_64_nodes_32,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
plt.xlabel('Training samples')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,100))
plt.savefig('No of training sample time_25.png')