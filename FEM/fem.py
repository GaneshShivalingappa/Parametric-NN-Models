import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import FunctionSpace, Function, locate_dofs_geometrical, dirichletbc, Constant, form
from ufl import TestFunction, TrialFunction, inner, dx, ds, nabla_grad
from petsc4py import PETSc
import scipy
from scipy.sparse import csr_matrix

def FEM(length:float, no_element:int, E:float, T:float):

    # Define physical variables
    L = length        # Length of the bar

    # Create the mesh
    num_elem = no_element  # Number of elements
    bar_mesh = mesh.create_interval(MPI.COMM_WORLD, num_elem, [0.0, L])

    # Define function space
    V = FunctionSpace(bar_mesh, ("Lagrange", 1))

    # Define boundary condition
    def on_boundary_L(x):								# Left edge to fix
        return np.isclose(x[0],0.0)

    dof_left_u = locate_dofs_geometrical(V, on_boundary_L)

    # Create Dirichlet boundary condition
    bc = dirichletbc(Constant(bar_mesh,ScalarType(0.0)), dof_left_u, V)
    #bc = fem.dirichletbc(u_D, dof_boundary)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define gravitational force
    f = Constant(bar_mesh, ScalarType(0.0))
    T = Constant(bar_mesh,ScalarType(T))
    E = Constant(bar_mesh,E)  

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
    #print(uh.x.array.real)
    #A = assemble_matrix(form(a))
    A = fem.petsc.assemble_matrix(form(a),bcs=[bc])
    #A = fem.petsc.assemble_matrix(form(a))
    A.assemble()
    f = fem.petsc.assemble_vector(form(L))
    fem.petsc.set_bc(f, [bc])
    f.assemble()
    #dolfinx.fem.petsc.apply_lifting_nest(A, f, [bc])
    ai, aj, av = A.getValuesCSR()
    Asp = csr_matrix((av, aj, ai))
    #print(Asp)
    #fem.petsc.apply_lifting(f, [a], [bc])
    #f.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(f, [bc])
    fnp = f.getArray()
    mat = scipy.sparse.csr_matrix.toarray(Asp)
    #np.save('stiff_mat.npy', mat)
    #np.save('force_vec.npy', fnp)
    return mat,fnp

def NN_FEM(length:float, no_element:int, E:float, T:float):

    # Define physical variables
    L = length        # Length of the bar

    # Create the mesh
    num_elem = no_element  # Number of elements
    bar_mesh = mesh.create_interval(MPI.COMM_WORLD, num_elem, [0.0, L])

    # Define function space
    V = FunctionSpace(bar_mesh, ("Lagrange", 1))

    # Define boundary condition
    def on_boundary_L(x):								# Left edge to fix
        return np.isclose(x[0],0.0)

    dof_left_u = locate_dofs_geometrical(V, on_boundary_L)

    # Create Dirichlet boundary condition
    bc = dirichletbc(Constant(bar_mesh,ScalarType(0.0)), dof_left_u, V)
    #bc = fem.dirichletbc(u_D, dof_boundary)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define gravitational force
    f = Constant(bar_mesh, ScalarType(0.0))
    T = Constant(bar_mesh,ScalarType(T))
    E = Constant(bar_mesh,E)  

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
    u = uh.x.array.real
    return u