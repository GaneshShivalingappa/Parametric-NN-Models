import matplotlib.pyplot as plt
FENN_mean_10_64_node_32 = 
FENN_std_10_64_node_32 = 

FENN_mean_10_64_training_sample_32 = [24.34754068851471, 21.424474120140076, 29.616368103027344, 32.20407762527466, 36.28398013114929, 36.297883653640746, 52.6488002538681, 57.08681902885437, 61.95816013813019, 65.24471971988677]
FENN_std_10_64_training_sample_32 = [6.085176831022796, 2.5935286698756963, 5.256660717249047, 4.27700892530613, 7.164968044370332, 5.862959554312527, 8.380669167659343, 9.376997391749917, 6.083360031094009,6.962406546005311]

time = [10,20,30,40,50,60,70,80,90,100]

PINN_mean_10_64_training_sample_32 = [4.6194456815719604, 4.427195715904236, 5.396327996253968, 4.547799921035766, 5.692430567741394, 6.214226865768433, 8.633271670341491, 9.366909193992615, 12.927103996276855, 12.60294713973999]
PINN_std_10_64_training_sample_32 = [1.167930413086073, 1.078304332124536, 1.3357562484001946, 1.2925540601747014, 1.4534376275556404, 1.244383143298019, 0.9589574842245824, 2.1825279136127618, 4.6768466924844345, 3.7503470809068786]

PINN_mean_10_64_node_32 = [4.038727617263794, 4.362580966949463, 4.884674048423767, 4.666429471969605, 5.195632910728454, 6.240629029273987, 9.122796416282654, 9.689197373390197, 8.867991185188293, 10.346008586883546]
PINN_std_10_64_node_32 = [0.5976190618943349, 1.350830108165585, 0.8388386214611412, 1.0490497366127747, 1.769604813712691, 1.9129695429255704, 1.9782611445763087, 2.24286927092742, 2.7789905908681543, 2.4209294489935296]

FEM_data_mean_10_64_nodes_32 = [2.7602012872695925, 4.480445337295532, 7.81294481754303, 13.58331778049469, 11.550549006462097, 14.294814944267273, 22.581231212615968, 31.953556299209595, 27.94034595489502, 23.085070991516112]
FEM_data_std_10_64_nodes_32  = [1.52232666497864, 4.390688324086366, 7.727011725548075, 12.106148712628006, 10.807345915670462, 14.885743303808114, 16.897745939494477, 26.45896395532895, 29.77513970841981, 10.974285131153962]

FEM_data_mean_10_64_training_sample_32 = [3.543100190162659, 6.8983272314071655, 6.815121459960937, 8.566356587409974, 14.634560418128967, 19.13566529750824, 28.62210876941681, 26.60739231109619, 51.505395126342776, 24.33567581176758]
FEM_data_std_10_64_training_sample_32 = [3.393671783164054, 6.70629955698704, 4.595983653761113, 3.5362081716903138, 13.340312702084422, 18.314343478907954, 16.05247473650931, 22.411350872732484, 39.09308768150033, 16.559035166131128]

plt.figure(1)
#plt.errorbar(time,FENN_mean_5_64_training_sample,FENN_std_5_64_training_sample,label='FEM-NN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FENN_mean_10_64_training_sample_32,FENN_std_10_64_training_sample_32,label='FEM-NN', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,PINN_mean_5_64_training_sample,PINN_std_5_64_training_sample,label='PINN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,PINN_mean_10_64_training_sample_32,PINN_std_10_64_training_sample_32,label='PINN', fmt='-o',capsize=3, linewidth=1)
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
plt.errorbar(time,FENN_mean_10_64_node_32,FENN_std_10_64_node_32,label='FEM-NN', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,PINN_mean_5_64_node,PINN_std_5_64_node,label='PINN_5', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,PINN_mean_10_64_node_32,PINN_std_10_64_node_32,label='PINN', fmt='-o',capsize=3, linewidth=1)
plt.errorbar(time,FEM_data_mean_10_64_nodes_32,FEM_data_std_10_64_nodes_32,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
#plt.errorbar(time,FEM_data_mean_10_64_nodes_50,FEM_data_std_10_64_nodes_50,label='FEM-Data', fmt='-o',capsize=3, linewidth=1)
#plt.title('Training time comparison (number of node = 64)')
plt.xlabel('Training samples')
plt.ylabel('Training time [s]')
plt.legend(loc='upper left')
#plt.ylim((None,100))
plt.savefig('No of training sample time.png')