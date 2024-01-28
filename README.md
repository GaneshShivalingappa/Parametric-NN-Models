# Introduction

In this project, we conducted a comparative analysis between the Physics Informed Neural Network (PINN) and the Finite Element Method enhanced neural network (FEM-NN), along with a FEM data-based neural network (FEM-Data-NN) models in the context of material parameter identification. These models can be used as surrogate models for material parameter identification by solving inverse problems. For this study, we considered a 1D bar fixed at one end and applied traction at the free end. Within this framework, we systematically evaluated and compared the performance of PINN, FEM-NN, and the FEM-Data-NN. The results were benchmarked against the analytical solution, and conducted a comprehensive training time analysis to provide insight into the computational efficiency of these methods.

# How it works

In this project we have performed two analysis.

1. Performance Analysis
2. Training Time Analysis

The container can be built by running the follwing command in the terminal.
```
singularity build --fakeroot --force parametric_nn.sif app/Singularity/container_production.def
```

## 1. Performance Analysis

We are comparing the displacement predicted from all three methods with the analytical solution. In addition, the relative error is calculated and plotted for each method. The plots for each method can be recreated in your local environment after creating the container by running the following command.

Parametric PINN 
```
singularity run --app Parametric_PINN parametric_nn.sif
```

Parametric FEM-NN
```
singularity run --app Parametric_FEM-NN parametric_nn.sif
```

Parametric FEM-data based model
```
singularity run --app Parametric_FEM-Data parametric_nn.sif
```
## 2. Training Time Analysis

We conducted training time analysis in two scenarios. In the first scenario, we kept the number of nodes constant by varying the number of samples, and in the second scenario, it is vice versa. The command to run the code to create the plot is given below.

Number of nodes varying while keeping the number of samples constant.
```
singularity run --app training_time_node parametric_nn.sif
```

Number of samples varying while keeping the number of nodes constant.
```
singularity run --app training_time_sample parametric_nn.sif
```

