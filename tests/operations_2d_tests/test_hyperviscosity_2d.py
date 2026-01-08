from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.mesh_generation.element_local_metric import create_quasi_uniform_grid_elem_local, create_mobius_like_grid_elem_local
from pysces.operations_2d.se_grid import init_hypervis_tensor, postprocess_grid
from pysces.operations_2d.operators import sphere_laplacian_wk, inner_prod
from pysces.operations_2d.assembly import project_scalar
from pysces.config import np, jnp, device_unwrapper, device_wrapper
from ..context import test_npts, get_figdir
from .tensor_hypervis_ref import tensor_hypervis_ref
from scipy.special import sph_harm_y


def test_hypervisc_tensor():
  pass
  for nx in [5, 6]:
    for npt in test_npts:
      grid_equi, _ = create_quasi_uniform_grid(nx, npt, wrapped=False)
      grid_elem_local, _ = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for grid in [grid_equi, grid_elem_local]:
        visc_tensor_for = tensor_hypervis_ref(grid["met_inv"], grid["jacobian_inv"])
        visc_tensor_operational = init_hypervis_tensor(grid["met_inv"], grid["jacobian_inv"])
        assert np.allclose(visc_tensor_for, visc_tensor_operational)


def test_hyperviscosity_sphere_harmonics():
  nx = 31
  radius_earth = 6371e3
  npt = 4
  nu_const = 1e15
  nu_tensor = 3.4e-8
  grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  grid = postprocess_grid(grid, dims)

  # need to check nu_tensor = 3.4e-8 with tensor HV leads to identical scaling as nu_const = 1e15 on ne30 grid
  # evaluate that hv_conversion = ( (np-1)*dx_unit_sphere / 2 )^{hv_scaling} * rearth^4
  # gives nu_tensor = nu_const * OPERATOR_HV^{-1}

  grid_nowrapper, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  lat = grid_nowrapper["physical_coords"][:, :, :, 0]
  lon = grid_nowrapper["physical_coords"][:, :, :, 1]
  m = 5
  l = 10
  Ymn = jnp.real(device_wrapper(sph_harm_y(l, m, lat + np.pi/2.0, lon)))
  laplace_Ymn_discont = sphere_laplacian_wk(Ymn, grid, a=1.0)
  laplace_Ymn = project_scalar(laplace_Ymn_discont, grid, dims, scaled=False)
  # check that we can resolve our spherical harmonic.
  diff = device_unwrapper(laplace_Ymn) - device_unwrapper(-l * (l+1) *  Ymn)

  biharmonic_Ymn_discont = sphere_laplacian_wk(-l * (l+1) *  Ymn, grid, a=1.0)
  biharmonic_Ymn = project_scalar(biharmonic_Ymn_discont, grid, dims, scaled=False)
  norm_const = (l * (l+1))**2

  import matplotlib.pyplot as plt
  plt.figure()
  plt.tricontourf(lon.flatten(), lat.flatten(), (biharmonic_Ymn /(norm_const * Ymn)).flatten(), levels=np.arange(0, 4.0, 0.5))
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/biharm_ratio.pdf")
  
  biharmonic_Ymn_discont_tensor = sphere_laplacian_wk(-l * (l+1) *  Ymn, grid, a=1.0, apply_tensor=True)
  biharmonic_Ymn_tensor = project_scalar(biharmonic_Ymn_discont_tensor, grid, dims, scaled=False)
  plt.figure()
  plt.tricontourf(lon[:, 1:-1, 1:-1].flatten(), lat[:, 1:-1, 1:-1].flatten(), (biharmonic_Ymn_tensor/norm_const)[:, 1:-1, 1:-1].flatten())
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/biharm_tensor.pdf")
  plt.figure()
  plt.tricontourf(lon[:, 1:-1, 1:-1].flatten(), lat[:, 1:-1, 1:-1].flatten(), (biharmonic_Ymn/norm_const)[:, 1:-1, 1:-1].flatten(), levels=np.linspace(-1, 1, 10))
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/biharm.pdf")

  # TODO [scalars]
  # * Write down what correct scaling and order of application is based on `nu` as
  # it is defined in standard SE dycores for scalars
  # * test hypothesized stability condition on earth-sized sphere
  # * Hypothesize choice of configuration that should render default hyperviscosity and 
  # tensor hyperviscosity approximately equivalent on grid.
  # TODO [Vectors]
      


def test_hyperviscosity_stability():
  radius_earth = 6371e3
  nx = 30
  npt = 4
  nu_const = 1e15
  grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  grid_nowrapper, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  lat = grid_nowrapper["physical_coords"][:, :, :, 0]
  lon = grid_nowrapper["physical_coords"][:, :, :, 1]
  for m in range(5):
    for l in range(90, 95):
      Ymn = jnp.real(device_wrapper(sph_harm_y(l, m, lat + np.pi/2.0, lon)))
      laplace_Ymn_discont = sphere_laplacian_wk(Ymn, grid, a=radius_earth)
      laplace_Ymn = project_scalar(laplace_Ymn_discont, grid, dims)

