import numpy as np
from numpy import linalg
import ufl
import dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import FunctionSpace, Function, locate_dofs_geometrical, dirichletbc, Constant, form
from ufl import TestFunction, TrialFunction, grad, inner, dx, ds, nabla_grad
#from dolfinx.io import XDMFFile
#from dolfinx.fem.assemble import assemble_matrix
import scipy
from scipy.sparse import csr_matrix

# Define physical variables
L = 1.0        # Length of the bar

# Create the mesh
num_elem = 5  # Number of elements
mesh = mesh.create_interval(MPI.COMM_WORLD, num_elem, [0.0, L])

# Define function space
V = FunctionSpace(mesh, ("Lagrange", 1))

# Define boundary condition
def on_boundary_L(x):								# Left edge to fix
    return np.isclose(x[0],0.0)

dof_left_u = locate_dofs_geometrical(V, on_boundary_L)

# Create Dirichlet boundary condition
bc = dirichletbc(Constant(mesh,0.0), dof_left_u, V)
#bc = fem.dirichletbc(u_D, dof_boundary)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define gravitational force
f = Constant(mesh, ScalarType(0.0))
T = Constant(mesh,ScalarType(5.0))
E = Constant(mesh,210.0)  

# Define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u))

def sigma(u):
    return E * epsilon(u)

# Define variational problem
a = inner(sigma(u), epsilon(v)) * dx
L = inner(f, v) * dx + ufl.inner(T, v) * ds
uh = Function(V)
# Create linear problem and solve
problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()
print(uh.x.array.real)
A = fem.petsc.assemble_matrix(form(a),bcs=[bc])
#A = fem.petsc.assemble_matrix(form(a))
A.assemble()

f = fem.petsc.assemble_vector(form(L))
fem.petsc.set_bc(f, [bc])
f.assemble()
ai, aj, av = A.getValuesCSR()
Asp = csr_matrix((av, aj, ai))
fnp = f.getArray()
mat = scipy.sparse.csr_matrix.toarray(Asp)
print(mat)
#np.save('stiff_mat.npy', mat)
#np.save('force_vec.npy', fnp)
print(fnp)

u = linalg.solve(mat,fnp)
print("np sol:\n", u)
'''import matplotlib.pyplot as plt
cells, types, x = plot.create_vtk_mesh(V)
plt.figure(1,dpi=300)  
exact = (2 * 5.0 * x[:,0] ) / (2 * 210)
plt.plot ( x[:,0], uh.x.array.real, label='FEM')
plt.plot ( x[:,0], exact, label='Analytical')
filename = 'only_traction_force.png'
plt.xlabel('Length')
plt.ylabel('displacement')
plt.title('1d Bar')
plt.legend()
plt.savefig ( filename )
print ( 'Graphics saved as "%s"' % ( filename ) )
plt.close ()'''

'''with XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)
import meshio

# Load the mesh from the MSH file
mesh = meshio.read("1d_bar.msh")

# Define the output filename for the XDMF file
output_filename = "1d_bar.xdmf"

# Write the mesh to the XDMF file
meshio.write(output_filename, mesh, file_format="xdmf")'''