# Introduction

In this project, we conducted a comparative analysis between the Physics Informed Neural Network (PINN) and the Finite Element Method enhanced neural network (FEM-NN), along with a FEM data-based model in the context of material parameter identification. These models can be surrogate models for material parameter identification by solving inverse problems. For this study, I considered a 1D bar fixed at one end and applied traction at the free end. Within this framework, I systematically evaluated and compared the performance of PINN, FEM-NN, and the FEM data-based model. The results were benchmarked against the analytical solution, and I conducted a comprehensive training time analysis to provide insight into the computational efficiency of these methods.

# How it works

In this project we have conducted two analysis on Physics Informed Neural Network, Finite Element Method enhanced neural network, FEM data-based model.

The container can be built by running follwing command in the terminal.
```
singularity build --fakeroot --force parametric_nn.sif app/Singularity/container_production.def
```

## 1. Performance Analysis

We are comparing the displacement predicted from all the three methods with the analytical solution. In addition the relative error is calculated and ploted for each method. The plotes for each method can be recreated in your local enviroment after creation the container by runing following command..

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




