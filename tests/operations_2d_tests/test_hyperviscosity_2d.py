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
  for nx in [5, 6]:
    for npt in test_npts:
      grid_equi, _ = create_quasi_uniform_grid(nx, npt, wrapped=False)
      grid_elem_local, _ = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for grid in [grid_equi, grid_elem_local]:
        visc_tensor_for = tensor_hypervis_ref(grid["met_inv"], grid["jacobian"])
        visc_tensor_operational = init_hypervis_tensor(grid["met_inv"], grid["jacobian"])
        assert np.allclose(visc_tensor_for, visc_tensor_operational)

def test_hypervisc_tensor_algebraic():
  nx = 31
  npt = 4
  grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  evals, evecs = np.linalg.eigh(grid["met_inv"])
  visc_tensor = init_hypervis_tensor(grid["met_inv"], grid["jacobian"], hypervis_scaling=0.0)
  shucked_tensor = np.einsum("fijsr,fijmr->fijsm", visc_tensor, grid["jacobian_inv"])
  shucked_tensor = np.einsum("fijsm,fijns->fijmn", shucked_tensor, grid["jacobian_inv"])
  assert jnp.max(jnp.abs(shucked_tensor - grid["met_inv"])) < 1e-8

  shucked_tensor = np.einsum("fijnm,fijnc->fijmc", shucked_tensor, evecs)
  shucked_tensor = np.einsum("fijmc,fijmd->fijdc", shucked_tensor, evecs)
  diag_evals = np.zeros_like(shucked_tensor)
  diag_evals[:, :, :, 0, 0] = evals[:, :, :, 0]
  diag_evals[:, :, :, 1, 1] = evals[:, :, :, 1]
  assert jnp.max(jnp.abs(diag_evals - shucked_tensor)) < 1e-8


def test_hyperviscosity_sphere_harmonics_uniform():
  nx = 31
  radius_earth = 6371e3
  npt = 4
  nu_const = 1e15
  nu_tensor = 2*3.4e-8
  hv_scaling = 3.2
  grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  #grid = postprocess_grid(grid, dims)

  # need to check nu_tensor = 3.4e-8 with tensor HV leads to identical scaling as nu_const = 1e15 on ne30 grid
  # evaluate that hv_conversion = ( (np-1)*dx_unit_sphere / 2 )^{hv_scaling} * rearth^4
  # gives nu_tensor = nu_const * OPERATOR_HV^{-1}

  grid_nowrapper, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  m = 5
  l = 10
  Ymn = jnp.real(device_wrapper(sph_harm_y(l, m, lat + np.pi/2.0, lon)))
  laplace_Ymn_discont = sphere_laplacian_wk(Ymn, grid, a=radius_earth)
  laplace_Ymn = project_scalar(laplace_Ymn_discont, grid, dims, scaled=False)
  # check that we can resolve our spherical harmonic.
  diff = device_unwrapper(laplace_Ymn) - device_unwrapper(-l * (l+1) *  Ymn)

  biharmonic_Ymn_discont = sphere_laplacian_wk(-l * (l+1) *  Ymn, grid, a=radius_earth)
  biharmonic_Ymn = project_scalar(biharmonic_Ymn_discont, grid, dims, scaled=False)
  norm_const = (l * (l+1))**2
  evals, evecs = np.linalg.eigh(grid["met_inv"])
  visc_tensor = init_hypervis_tensor(grid["met_inv"], grid["jacobian"], hypervis_scaling=hv_scaling)
  shucked_tensor = np.einsum("fijsr,fijmr->fijsm", visc_tensor, grid["jacobian_inv"])
  shucked_tensor = np.einsum("fijsm,fijns->fijmn", shucked_tensor, grid["jacobian_inv"])

  shucked_tensor = np.einsum("fijnm,fijnc->fijmc", shucked_tensor, evecs)
  shucked_tensor = np.einsum("fijmc,fijmd->fijdc", shucked_tensor, evecs)

  lamStar1 = shucked_tensor[:, :, :, 0, 0]/evals[:, :, :, 0]
  lamStar2 = shucked_tensor[:, :, :, 1, 1]/evals[:, :, :, 1]
  # todo: reconstruct hv_conversion so this is actually a test.

  hv_conversion = np.sqrt(evals[:, :, :, 0])**(-hv_scaling) * radius_earth**4
  hv_conversion_2 = lamStar1 * radius_earth**4
  assert np.allclose(hv_conversion, hv_conversion_2)
  hv_conversion = np.sqrt(evals[:, :, :, 1])**(-hv_scaling) * radius_earth**4
  hv_conversion_2 = lamStar2 * radius_earth**4
  assert np.allclose(hv_conversion, hv_conversion_2)

  biharmonic_Ymn_discont_tensor = sphere_laplacian_wk(-l * (l+1) *  Ymn, grid, a=1.0, apply_tensor=True)
  biharmonic_Ymn_tensor = project_scalar(biharmonic_Ymn_discont_tensor, grid, dims, scaled=False)

  maybe_equivalent_scaling = (nu_tensor * hv_conversion) * biharmonic_Ymn/norm_const
  reference_scaling = nu_const * biharmonic_Ymn/norm_const
  reference_scaling_tensor = nu_tensor * biharmonic_Ymn_tensor/norm_const

  # print(jnp.abs(jnp.log(jnp.abs(jnp.max(maybe_equivalent_scaling))) - jnp.log(jnp.abs(jnp.max(reference_scaling)))))
  # print(jnp.abs(jnp.log(jnp.abs(jnp.min(maybe_equivalent_scaling))) - jnp.log(jnp.abs(jnp.min(reference_scaling)))))
  # print(jnp.max(reference_scaling_tensor))
  # print(jnp.min(reference_scaling_tensor))

  # * Write down what correct scaling and order of application is based on `nu` as
  # it is defined in standard SE dycores for scalars
  # * test hypothesized stability condition on earth-sized sphere
  # * Hypothesize choice of configuration that should render default hyperviscosity and 
  # tensor hyperviscosity approximately equivalent on grid.
  # TODO [Vectors]


def test_hyperviscosity_sphere_harmonics_mobius():
  nx = 31
  radius_earth = 6371e3
  npt = 4
  nu_const = 1e15
  nu_tensor =  3.4e-8
  grid_squish, dims_sqish = create_mobius_like_grid_elem_local(nx, npt, axis_dilation=np.array([1.0, 2.0, 1.0]))
  grid_uniform, dims = create_quasi_uniform_grid_elem_local(nx, npt)

  for grid, label in zip([grid_uniform, grid_squish], ["uniform", "squish"]):
    grid = postprocess_grid(grid, dims)

    # need to check nu_tensor = 3.4e-8 with tensor HV leads to identical scaling as nu_const = 1e15 on ne30 grid
    # evaluate that hv_conversion = ( (np-1)*dx_unit_sphere / 2 )^{hv_scaling} * rearth^4
    # gives nu_tensor = nu_const * OPERATOR_HV^{-1}

    grid_nowrapper, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
    lat = grid["physical_coords"][:, :, :, 0]
    lon = grid["physical_coords"][:, :, :, 1]
    m = 5
    l = 10
    Ymn = jnp.real(device_wrapper(sph_harm_y(l, m, lat + np.pi/2.0, lon)))
    laplace_Ymn_discont = sphere_laplacian_wk(Ymn, grid, a=radius_earth)
    laplace_Ymn = project_scalar(laplace_Ymn_discont, grid, dims, scaled=False)
    # check that we can resolve our spherical harmonic.
    diff = device_unwrapper(laplace_Ymn) - device_unwrapper(-l * (l+1) *  Ymn)

    biharmonic_Ymn_discont = sphere_laplacian_wk(-l * (l+1) *  Ymn, grid, a=radius_earth)
    biharmonic_Ymn = project_scalar(biharmonic_Ymn_discont, grid, dims, scaled=False)
    norm_const = (l * (l+1))**2
    evals, evecs = np.linalg.eigh(grid["met_inv"])
    hv_scaling = 3.2
    hv_conversion = np.sqrt(evals[:, :, :, 0])**(-hv_scaling) * radius_earth**4
    ne_30_mean = 4 * np.pi / grid["met_det"].shape[0]
    per_element_area_grid = np.sum(grid["met_det"] *
                                   grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                   grid["gll_weights"][np.newaxis, np.newaxis, :], axis=(1, 2))
    area_ratio = grid["mass_mat"] / grid_uniform["mass_mat"]
    variable_resolution_coefficient = np.sqrt(area_ratio)**(hv_scaling)

    import matplotlib.pyplot as plt

    biharmonic_Ymn_discont_tensor = sphere_laplacian_wk(-l * (l+1) *  Ymn, grid, a=radius_earth, apply_tensor=True)
    biharmonic_Ymn_tensor = project_scalar(biharmonic_Ymn_discont_tensor, grid, dims, scaled=False)
    reference_scaling = nu_const * biharmonic_Ymn/norm_const
    reference_scaling_tensor = nu_tensor * biharmonic_Ymn_tensor/norm_const / variable_resolution_coefficient
    
    mask = np.logical_and(jnp.abs(reference_scaling_tensor) > jnp.max(jnp.abs(reference_scaling_tensor))/2,
                          jnp.abs(reference_scaling) > jnp.max(jnp.abs(reference_scaling))/2)
    ratio = np.log10(np.abs(reference_scaling_tensor[mask] / reference_scaling[mask]))
    assert np.max(np.abs(ratio)) < 1.0

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
