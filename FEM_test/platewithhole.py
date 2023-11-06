import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeAlias, Union

import dolfinx
import gmsh
import numpy as np
import numpy.typing as npt
import pandas as pd
import petsc4py
import ufl
from dolfinx.fem import Constant, FunctionSpace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import Mesh, locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorElement,
    as_matrix,
    as_tensor,
    as_vector,
    dot,
    dx,
    inner,
    inv,
    lhs,
    nabla_grad,
    rhs,
)

from parametricpinn.errors import FEMConfigurationError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import DataclassWriter, PandasDataWriter
from parametricpinn.settings import Settings
from parametricpinn.types import NPArray

GMesh: TypeAlias = gmsh.model
GOutDimAndTags: TypeAlias = list[tuple[int, int]]
GOutDimAndTagsMap: TypeAlias = list[Union[GOutDimAndTags, list[Any]]]
GGeometry: TypeAlias = tuple[GOutDimAndTags, GOutDimAndTagsMap]
DMesh: TypeAlias = dolfinx.mesh.Mesh
DFunction: TypeAlias = dolfinx.fem.Function
DFunctionSpace: TypeAlias = dolfinx.fem.FunctionSpace
DTestFunction: TypeAlias = ufl.TestFunction
DConstant: TypeAlias = dolfinx.fem.Constant
DDofs: TypeAlias = npt.NDArray[np.int32]
DMeshTags: TypeAlias = Any  # dolfinx.mesh.MeshTags
DDirichletBC: TypeAlias = dolfinx.fem.DirichletBCMetaClass
UFLOperator: TypeAlias = ufl.core.operator.Operator
UFLMeasure: TypeAlias = ufl.Measure
UFLSigmaFunc: TypeAlias = Callable[[TrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[TrialFunction], UFLOperator]
PETScScalarType: TypeAlias = petsc4py.PETSc.ScalarType


@dataclass
class PWHSimulationConfig:
    model: str
    youngs_modulus: float
    poissons_ratio: float
    edge_length: float
    radius: float
    volume_force_x: float
    volume_force_y: float
    traction_left_x: float
    traction_left_y: float
    element_family: str
    element_degree: int
    mesh_resolution: float


Config: TypeAlias = PWHSimulationConfig
BCValue: TypeAlias = Union[DConstant, PETScScalarType]


@dataclass
class SimulationResults:
    coordinates_x: NPArray
    coordinates_y: NPArray
    youngs_modulus: float
    poissons_ratio: float
    displacements_x: NPArray
    displacements_y: NPArray


geometric_dim = 2


def run_simulation(
    model: str,
    youngs_modulus: float,
    poissons_ratio: float,
    edge_length: float,
    radius: float,
    volume_force_x: float,
    volume_force_y: float,
    traction_left_x: float,
    traction_left_y: float,
    save_results: bool,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    element_family: str = "Lagrange",
    element_degree: int = 1,
    mesh_resolution: float = 1,
) -> SimulationResults:
    simulation_config = PWHSimulationConfig(
        model=model,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        edge_length=edge_length,
        radius=radius,
        volume_force_x=volume_force_x,
        volume_force_y=volume_force_y,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        element_family=element_family,
        element_degree=element_degree,
        mesh_resolution=mesh_resolution,
    )
    mesh = _generate_mesh(
        simulation_config, save_results, output_subdir, project_directory
    )
    simulation_results = _simulate_once(
        mesh, simulation_config, save_metadata, output_subdir, project_directory
    )
    if save_results:
        _save_results(
            simulation_results, simulation_config, output_subdir, project_directory
        )
    return simulation_results


def generate_validation_data(
    model: str,
    youngs_moduli: list[float],
    poissons_ratios: list[float],
    edge_length: float,
    radius: float,
    volume_force_x: float,
    volume_force_y: float,
    traction_left_x: float,
    traction_left_y: float,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    element_family: str = "Lagrange",
    element_degree: int = 1,
    mesh_resolution: float = 1,
) -> None:
    save_results = True
    save_to_input_dir = True
    simulation_config = PWHSimulationConfig(
        model=model,
        youngs_modulus=0.0,
        poissons_ratio=0.0,
        edge_length=edge_length,
        radius=radius,
        volume_force_x=volume_force_x,
        volume_force_y=volume_force_y,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        element_family=element_family,
        element_degree=element_degree,
        mesh_resolution=mesh_resolution,
    )
    num_simulations = _determine_number_of_simulations(youngs_moduli, poissons_ratios)
    mesh = _generate_mesh(
        simulation_config,
        save_results,
        output_subdir,
        project_directory,
        save_to_input_dir=save_to_input_dir,
    )

    for simulation_count, (youngs_modulus, poissons_ratio) in enumerate(
        zip(youngs_moduli, poissons_ratios)
    ):
        print(f"Run FEM simulation {simulation_count + 1}/{num_simulations} ...")
        simulation_config.youngs_modulus = youngs_modulus
        simulation_config.poissons_ratio = poissons_ratio
        simulation_name = f"sample_{simulation_count}"
        simulation_output_subdir = _join_simulation_output_subdir(
            simulation_name, output_subdir
        )
        simulation_results = _simulate_once(
            mesh,
            simulation_config,
            save_metadata,
            simulation_output_subdir,
            project_directory,
            save_to_input_dir=save_to_input_dir,
        )
        _save_results(
            simulation_results,
            simulation_config,
            simulation_output_subdir,
            project_directory,
            save_to_input_dir=save_to_input_dir,
        )


def _determine_number_of_simulations(
    youngs_moduli: list[float], poissons_ratios: list[float]
) -> int:
    if not len(youngs_moduli) == len(poissons_ratios):
        raise FEMConfigurationError(
            f"Not the same number of Young's moduli and Poissons ratios."
        )
    return len(youngs_moduli)


def _join_simulation_output_subdir(simulation_name: str, output_subdir: str) -> str:
    return os.path.join(output_subdir, simulation_name)


def _generate_mesh(
    config: Config,
    save_mesh: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> DMesh:
    print("Generate mesh for FEM simulation ...")
    gmsh.initialize()
    gmesh = _generate_gmesh(config)
    if save_mesh:
        _save_gmesh(output_subdir, save_to_input_dir, project_directory)
    mesh = _load_mesh_from_gmsh_model(gmesh)
    gmsh.finalize()
    return mesh


def _generate_gmesh(config: Config) -> GMesh:
    length = config.edge_length
    radius = config.radius
    resolution = config.mesh_resolution
    geometry_kernel = gmsh.model.occ
    solid_marker = 1

    def create_geometry() -> GGeometry:
        gmsh.model.add("domain")
        plate = geometry_kernel.add_rectangle(0, 0, 0, -length, length)
        hole = geometry_kernel.add_disk(0, 0, 0, radius, radius)
        return geometry_kernel.cut([(2, plate)], [(2, hole)])

    def tag_physical_enteties(geometry: GGeometry) -> None:
        geometry_kernel.synchronize()

        def tag_solid_surface() -> None:
            surface = geometry_kernel.getEntities(dim=2)
            assert surface == geometry[0]
            gmsh.model.addPhysicalGroup(surface[0][0], [surface[0][1]], solid_marker)
            gmsh.model.setPhysicalName(surface[0][0], solid_marker, "Solid")

        tag_solid_surface()

    def configure_mesh() -> None:
        gmsh.model.mesh.setSizeCallback(mesh_size_callback)

    def mesh_size_callback(
        dim: int, tag: int, x: float, y: float, z: float, lc: float
    ) -> float:
        return resolution

    def generate_mesh() -> None:
        geometry_kernel.synchronize()
        gmsh.model.mesh.generate(geometric_dim)

    geometry = create_geometry()
    tag_physical_enteties(geometry)
    configure_mesh()
    generate_mesh()

    return gmsh.model


def _save_gmesh(
    output_subdir: str, save_to_input_dir: bool, project_directory: ProjectDirectory
) -> None:
    file_name = "mesh.msh"
    output_path = _join_output_path(
        project_directory, file_name, output_subdir, save_to_input_dir
    )
    gmsh.write(str(output_path))


def _load_mesh_from_gmsh_model(gmesh: GMesh) -> Mesh:
    mpi_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(
        gmesh, MPI.COMM_WORLD, mpi_rank, gdim=geometric_dim
    )
    return mesh


class NeumannBC:
    def __init__(
        self, tag: int, value: BCValue, measure: UFLMeasure, test_func: DTestFunction
    ) -> None:
        self.bc = inner(value, test_func) * measure(tag)


class DirichletBC:
    def __init__(
        self, dofs: DDofs, value: BCValue, dim: int, func_space: DFunctionSpace
    ) -> None:
        self.bc = dirichletbc(value, dofs, func_space.sub(dim))


BoundaryConditions: TypeAlias = list[Union[DirichletBC, NeumannBC]]


def _simulate_once(
    mesh: Mesh,
    config: Config,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> SimulationResults:
    model = config.model
    youngs_modulus = config.youngs_modulus
    poissons_ratio = config.poissons_ratio
    length = config.edge_length
    radius = config.radius
    volume_force_x = config.volume_force_x
    volume_force_y = config.volume_force_y
    traction_left_x = config.traction_left_x
    traction_left_y = config.traction_left_y
    traction_top_x = traction_top_y = 0.0
    traction_hole_x = traction_hole_y = 0.0
    element_family = config.element_family
    element_degree = config.element_degree

    T_top = Constant(mesh, (ScalarType((traction_top_x, traction_top_y))))
    T_hole = Constant(mesh, (ScalarType((traction_hole_x, traction_hole_y))))
    T_left = Constant(mesh, (ScalarType((traction_left_x, traction_left_y))))
    u_x_right = ScalarType(0.0)
    u_y_bottom = ScalarType(0.0)
    f = Constant(mesh, (ScalarType((volume_force_x, volume_force_y))))
    bc_facets_dim = mesh.topology.dim - 1

    tag_top = 0
    tag_right = 1
    tag_hole = 2
    tag_bottom = 3
    tag_left = 4
    locate_top_facet = lambda x: np.isclose(x[1], length)
    locate_right_facet = lambda x: np.isclose(x[0], 0.0)
    locate_hole_facet = lambda x: np.isclose(
        np.sqrt(np.square(x[0]) + np.square(x[1])), radius
    )
    locate_bottom_facet = lambda x: np.isclose(x[1], 0.0)
    locate_left_facet = lambda x: np.isclose(x[0], -length)

    element = VectorElement(element_family, mesh.ufl_cell(), element_degree)
    func_space = FunctionSpace(mesh, element)
    x = SpatialCoordinate(mesh)
    u = TrialFunction(func_space)
    w = TestFunction(func_space)

    def sigma_and_epsilon_factory() -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        compliance_matrix = None
        if model == "plane stress":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [1.0, -poissons_ratio, 0.0],
                    [-poissons_ratio, 1.0, 0.0],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
                ]
            )
        elif model == "plane strain":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [
                        1.0 - poissons_ratio**2,
                        -poissons_ratio * (1.0 + poissons_ratio),
                        0.0,
                    ],
                    [
                        -poissons_ratio * (1.0 + poissons_ratio),
                        1.0 - poissons_ratio**2,
                        0.0,
                    ],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
                ]
            )
        else:
            raise FEMConfigurationError(f"Unknown model: {model}")

        elasticity_matrix = inv(compliance_matrix)

        def sigma(u: TrialFunction) -> UFLOperator:
            return _sigma_voigt_to_matrix(
                dot(elasticity_matrix, _epsilon_matrix_to_voigt(epsilon(u)))
            )

        def _epsilon_matrix_to_voigt(eps: UFLOperator) -> UFLOperator:
            return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

        def _sigma_voigt_to_matrix(sig: UFLOperator) -> UFLOperator:
            return as_tensor([[sig[0], sig[2]], [sig[2], sig[1]]])

        def epsilon(u: TrialFunction) -> UFLOperator:
            return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

        return sigma, epsilon

    def tag_boundaries() -> DMeshTags:
        boundaries = [
            (tag_top, locate_top_facet),
            (tag_right, locate_right_facet),
            (tag_hole, locate_hole_facet),
            (tag_bottom, locate_bottom_facet),
            (tag_left, locate_left_facet),
        ]

        facet_indices_list: list[npt.NDArray[np.int32]] = []
        facet_tags_list: list[npt.NDArray[np.int32]] = []
        for tag, locator_func in boundaries:
            _facet_indices = locate_entities(mesh, bc_facets_dim, locator_func)
            facet_indices_list.append(_facet_indices)
            facet_tags_list.append(np.full_like(_facet_indices, tag))
        facet_indices = np.hstack(facet_indices_list).astype(np.int32)
        facet_tags = np.hstack(facet_tags_list).astype(np.int32)
        sorted_facet_indices = np.argsort(facet_indices)
        return meshtags(
            mesh,
            bc_facets_dim,
            facet_indices[sorted_facet_indices],
            facet_tags[sorted_facet_indices],
        )

    def define_boundary_conditions(
        boundary_tags: DMeshTags,
    ) -> BoundaryConditions:
        facet_right = boundary_tags.find(tag_right)
        dofs_right = locate_dofs_topological(func_space, bc_facets_dim, facet_right)
        facet_bottom = boundary_tags.find(tag_bottom)
        dofs_bottom = locate_dofs_topological(func_space, bc_facets_dim, facet_bottom)

        return [
            NeumannBC(tag=tag_top, value=T_top, measure=ds, test_func=w),
            DirichletBC(dofs=dofs_right, value=u_x_right, dim=0, func_space=func_space),
            NeumannBC(tag=tag_hole, value=T_hole, measure=ds, test_func=w),
            DirichletBC(
                dofs=dofs_bottom, value=u_y_bottom, dim=1, func_space=func_space
            ),
            NeumannBC(tag=tag_left, value=T_left, measure=ds, test_func=w),
        ]

    def apply_boundary_conditions(
        boundary_conditions: BoundaryConditions, F: UFLOperator
    ) -> tuple[list[DDirichletBC], UFLOperator]:
        dirichlet_bcs = []
        for condition in boundary_conditions:
            if isinstance(condition, DirichletBC):
                dirichlet_bcs.append(condition.bc)
            else:
                F += condition.bc
        return dirichlet_bcs, F

    def save_boundary_tags_as_xdmf(boundary_tags: DMeshTags) -> None:
        file_name = "boundary_tags.xdmf"
        output_path = _join_output_path(
            project_directory, file_name, output_subdir, save_to_input_dir
        )
        # mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        with XDMFFile(mesh.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(boundary_tags)

    def save_results_as_xdmf(mesh: DMesh, uh: DFunction) -> None:
        file_name = "displacements.xdmf"
        output_path = _join_output_path(
            project_directory, file_name, output_subdir, save_to_input_dir
        )
        with XDMFFile(mesh.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(uh)

    def compile_output(mesh: DMesh, uh: DFunction) -> SimulationResults:
        coordinates = mesh.geometry.x
        coordinates_x = coordinates[:, 0].reshape((-1, 1))
        coordinates_y = coordinates[:, 1].reshape((-1, 1))

        displacements = uh.x.array.reshape((-1, mesh.geometry.dim))
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))

        simulation_results = SimulationResults(
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
        )

        return simulation_results

    boundary_tags = tag_boundaries()

    ds = Measure("ds", domain=mesh, subdomain_data=boundary_tags)

    boundary_conditions = define_boundary_conditions(boundary_tags)
    sigma, epsilon = sigma_and_epsilon_factory()

    F = inner(sigma(u), epsilon(w)) * dx - inner(w, f) * dx

    dirichlet_bcs, F = apply_boundary_conditions(boundary_conditions, F)

    a = lhs(F)
    L = rhs(F)
    problem = LinearProblem(
        a, L, bcs=dirichlet_bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    if save_metadata:
        save_boundary_tags_as_xdmf(boundary_tags)
        save_results_as_xdmf(mesh, uh)

    return compile_output(mesh, uh)


def _save_results(
    simulation_results: SimulationResults,
    simulation_config: PWHSimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> None:
    _save_simulation_results(
        simulation_results, output_subdir, save_to_input_dir, project_directory
    )
    _save_simulation_config(
        simulation_config, output_subdir, save_to_input_dir, project_directory
    )


def _save_simulation_results(
    simulation_results: SimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    _save_displacements(
        simulation_results, output_subdir, save_to_input_dir, project_directory
    )
    _save_parameters(
        simulation_results, output_subdir, save_to_input_dir, project_directory
    )


def _save_displacements(
    simulation_results: SimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    data_writer = PandasDataWriter(project_directory)
    file_name = "displacements"
    results = simulation_results
    results_dict = {
        "coordinates_x": np.ravel(results.coordinates_x),
        "coordinates_y": np.ravel(results.coordinates_y),
        "displacements_x": np.ravel(results.displacements_x),
        "displacements_y": np.ravel(results.displacements_y),
    }
    results_dataframe = pd.DataFrame(results_dict)
    data_writer.write(
        results_dataframe,
        file_name,
        output_subdir,
        header=True,
        save_to_input_dir=save_to_input_dir,
    )


def _save_parameters(
    simulation_results: SimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    data_writer = PandasDataWriter(project_directory)
    file_name = "parameters"
    results = simulation_results
    results_dict = {
        "youngs_modulus": np.array([results.youngs_modulus]),
        "poissons_ratio": np.array([results.poissons_ratio]),
    }
    results_dataframe = pd.DataFrame(results_dict)
    data_writer.write(
        results_dataframe,
        file_name,
        output_subdir,
        header=True,
        save_to_input_dir=save_to_input_dir,
    )


def _save_simulation_config(
    simulation_config: PWHSimulationConfig,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    data_writer = DataclassWriter(project_directory)
    file_name = "simulation_config"
    data_writer.write(
        simulation_config, file_name, output_subdir, save_to_input_dir=save_to_input_dir
    )


def _join_output_path(
    project_directory: ProjectDirectory,
    file_name: str,
    output_subdir: str,
    save_to_input_dir: bool,
) -> Path:
    if save_to_input_dir:
        return project_directory.create_input_file_path(
            file_name=file_name, subdir_name=output_subdir
        )
    return project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
