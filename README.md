# Introduction

In this project, WE conducted a comparative analysis between the Physics Informed Neural Network (PINN) and the Finite Element Method enhanced neural network (FEM-NN), along with a FEM data-based model in the context of material parameter identification. These models can be surrogate models for material parameter identification by solving inverse problems. For this study, I considered a 1D bar fixed at one end and applied traction at the free end. Within this framework, I systematically evaluated and compared the performance of PINN, FEM-NN, and the FEM data-based model. The results were benchmarked against the analytical solution, and I conducted a comprehensive training time analysis to provide insight into the computational efficiency of these methods.

# How it works

In this project we are performing two analysis.

# 1. Performance Analysis

In this we are comparing the displacement predicted from all the three methods with the analytical solution. In addition the relative error is calculated and ploted for each method. these plotes can be recrealed by runing following command.

Parametric PINN
%singularity run --app Parametric_PINN my_container.sif

Parametric FEM-NN
%singularity run --app Parametric_FEM-NN my_container.sif

Parametric FEM-data based model
%singularity run --app Parametric_FEM-Data my_container.sif
