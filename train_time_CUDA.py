import matplotlib.pyplot as plt
FENN_mean_25_64_node_32 = [1.951828622817993, 3.9882031059265137, 5.688660078048706, 7.425619325637817, 10.22009539604187, 12.682736158370972, 14.911315097808838, 17.566327390670775, 20.735085639953613, 30.23120390892029]
FENN_std_25_64_node_32 = [0.26068021819282533, 0.5496730814091645, 0.7336747155477045, 0.8486619643471335, 1.3849925132003198, 1.3175873656047576, 2.3489877583818792, 2.6077083196759334, 2.7319680812599456, 6.6638681485358155]

FENN_mean_25_64_training_sample_32 = [2.4180569648742676, 3.840162878036499, 6.029633541107177, 7.610850620269775, 9.391145763397217, 12.1095445728302, 13.441783351898193, 16.660348024368286, 18.84737690925598, 22.1381458568573]
FENN_std_25_64_training_sample_32 = [0.5051065977119326, 0.5657168230382024, 1.0252880158017954, 1.1712238723893764, 1.189485876888535, 1.8623703294545175, 2.4631996766892423, 2.7165968357702943, 2.8338320139533466]

time = [10,20,30,40,50,60,70,80,90,100]

PINN_mean_25_64_training_sample_32 = [2.9869573402404783, 3.2734594345092773, 2.809400825500488, 2.943143014907837, 3.0929330825805663, 3.1253977489471434, 3.5827964782714843, 3.389816932678223, 3.872451229095459, 3.832998819351196]
PINN_std_25_64_training_sample_32 = [0.7192044745925901, 0.7341131254961213, 0.47337690597811993, 0.6118961013414689, 0.9181055534334738, 0.9710498306995001, 0.8212973892513926, 0.9127271477812957, 0.9053831549836127, 1.0831900017905738]

PINN_mean_25_64_node_32 = [2.213827724456787, 2.4760064601898195, 2.849748229980469, 2.6159934329986574, 3.073472013473511, 3.4130670738220217, 3.325650243759155, 3.6604987144470216, 3.7040756225585936, 3.663414182662964]
PINN_std_25_64_node_32 = [0.6117492833362412, 0.6335384855634466, 0.5869086926217358, 0.6570191923077201, 0.8729779651711693, 0.6758015222273265, 0.8559554785324527, 0.7751253801940063, 0.6546558746296426, 0.8769127650436898]

FEM_data_mean_10_64_nodes_32 = 
FEM_data_std_10_64_nodes_32  = 

FEM_data_mean_10_64_training_sample_32 = 
FEM_data_std_10_64_training_sample_32 = 

plt.figure(1)
#plt.errorbar(time,FENN_mean_5_64_training_sample,FENN_std_5_64_training_sample,label='FEM-NN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FENN_mean_25_64_training_sample_32,FENN_std_25_64_training_sample_32,label='FEM-NN', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,PINN_mean_5_64_training_sample,PINN_std_5_64_training_sample,label='PINN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,PINN_mean_25_64_training_sample_32,PINN_std_25_64_training_sample_32,label='PINN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FEM_data_mean_10_64_training_sample_32,FEM_data_std_10_64_training_sample_32,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,FEM_data_mean_10_64_training_sample_50,FEM_data_std_10_64_training_sample_50,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
#plt.title('Training time comparison (number of training sample = 64)')
plt.xlabel('Nodes')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,100))
plt.savefig('No of node time.png')
plt.figure(2)
#plt.errorbar(time,FENN_mean_5_64_node,FENN_std_5_64_node,label='FEM-NN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FENN_mean_25_64_node_32,FENN_std_25_64_node_32,label='FEM-NN', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,PINN_mean_5_64_node,PINN_std_5_64_node,label='PINN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,PINN_mean_25_64_node_32,PINN_std_25_64_node_32,label='PINN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FEM_data_mean_10_64_nodes_32,FEM_data_std_10_64_nodes_32,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,FEM_data_mean_10_64_nodes_50,FEM_data_std_10_64_nodes_50,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
#plt.title('Training time comparison (number of node = 64)')
plt.xlabel('Training samples')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,100))
plt.savefig('No of training sample time.png')