from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.mesh_generation.element_local_metric import create_quasi_uniform_grid_elem_local
from pysces.operations_2d.se_grid import init_hypervis_tensor
from pysces.config import np
from ..context import test_npts
from .tensor_hypervis_ref import tensor_hypervis_ref

def test_hypervisc_tensor():
  for nx in [5, 6]:
    for npt in test_npts:
      grid_equi, _ = create_quasi_uniform_grid(nx, npt, wrapped=False)
      grid_elem_local, _ = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for grid in [grid_equi, grid_elem_local]:
        visc_tensor_for = tensor_hypervis_ref(grid["met_inv"], grid["jacobian_inv"])
        visc_tensor_operational = init_hypervis_tensor(grid["met_inv"], grid["jacobian_inv"])
        assert np.allclose(visc_tensor_for, visc_tensor_operational)
