from ..config import np, device_wrapper, use_wrapper, device_unwrapper, jnp
from frozendict import frozendict
from .local_assembly import triage_vert_redundancy_flat, init_assembly_global, init_assembly_local
from ..mesh_generation.coordinate_utils import bilinear
from ..distributed_memory.global_assembly import project_scalar_global
from ..distributed_memory.global_communication import global_max, global_min
from .tensor_hyperviscosity import init_hypervis_tensor
from ..spectral import init_spectral


def subset_var(var, proc_idx, decomp, element_reordering=None, wrapped=use_wrapper):
  NELEM_GLOBAL = var.shape[0]
  if element_reordering is None:
    element_reordering = np.arange(0, NELEM_GLOBAL)
  dtype = var.dtype
  if wrapped:
    var_np = device_unwrapper(var)
  else:
    var_np = var
  var_subset = np.take(var_np, element_reordering[decomp[proc_idx][0]:decomp[proc_idx][1]], axis=0)
  if wrapped:
    var_out = device_wrapper(var_subset, dtype=dtype)
  else:
    var_out = var_subset
  return var_out


def create_spectral_element_grid(latlon,
                                 gll_to_sphere_jacobian,
                                 gll_to_sphere_jacobian_inv,
                                 physical_coords_to_cartesian,
                                 rmetdet,
                                 metdet,
                                 mass_mat,
                                 inv_mass_mat,
                                 vert_redundancy_gll_flat,
                                 proc_idx,
                                 decomp,
                                 element_reordering=None,
                                 wrapped=use_wrapper):

  # note: test code sometimes sets wrapped=False to test wrapper library (jax, torch) vs stock numpy
  # this extra conditional is not extraneous.
  if wrapped:
    wrapper = device_wrapper
  else:
    def wrapper(x, dtype=None):
      return x

  def subset_wrapper(field, dtype=None):
    return subset_var(wrapper(field, dtype=dtype), proc_idx, decomp,
                      element_reordering=element_reordering, wrapped=wrapped)

  NELEM = subset_wrapper(metdet).shape[0]
  npt = metdet.shape[1]
  # This function currently assumes that the full grid can be loaded into memory.
  # This should be fine up to, e.g., quarter-degree grids.

  spectrals = init_spectral(npt)
  vert_red_local, vert_red_send, vert_red_recv = triage_vert_redundancy_flat(vert_redundancy_gll_flat,
                                                                             proc_idx,
                                                                             decomp)
  assembly_triple = init_assembly_local(NELEM, npt, vert_red_local)
  triples_send, triples_recv = init_assembly_global(NELEM, npt, vert_red_send, vert_red_recv)

  met_inv = np.einsum("fijgs, fijhs->fijgh",
                      gll_to_sphere_jacobian_inv,
                      gll_to_sphere_jacobian_inv)
  mass_matrix = (metdet *
                 spectrals["gll_weights"][np.newaxis, :, np.newaxis] *
                 spectrals["gll_weights"][np.newaxis, np.newaxis, :])

  for proc_idx_recv in triples_recv.keys():
    triples_recv[proc_idx_recv] = (wrapper(triples_recv[proc_idx_recv][0]),
                                   wrapper(triples_recv[proc_idx_recv][1], dtype=jnp.int64),
                                   wrapper(triples_recv[proc_idx_recv][2], dtype=jnp.int64))

  for proc_idx_send in triples_send.keys():
    triples_send[proc_idx_send] = (wrapper(triples_send[proc_idx_send][0]),
                                   wrapper(triples_send[proc_idx_send][1], dtype=jnp.int64),
                                   wrapper(triples_send[proc_idx_send][2], dtype=jnp.int64))
  viscosity_tensor, hypervis_scaling = init_hypervis_tensor(met_inv, gll_to_sphere_jacobian)
  ret = {"physical_coords": subset_wrapper(latlon),
         "physical_to_cartesian": subset_wrapper(physical_coords_to_cartesian),
         "contra_to_physical": subset_wrapper(gll_to_sphere_jacobian),
         "physical_to_contra": subset_wrapper(gll_to_sphere_jacobian_inv),
         "recip_metric_determinant": subset_wrapper(rmetdet),
         "metric_determinant": subset_wrapper(metdet),
         "mass_matrix": subset_wrapper(mass_matrix),
         "mass_matrix_denominator": subset_wrapper(inv_mass_mat),
         "metric_inverse": subset_wrapper(met_inv),
         "viscosity_tensor": subset_wrapper(viscosity_tensor),
         "derivative_matrix": wrapper(spectrals["deriv"]),
         "gll_weights": wrapper(spectrals["gll_weights"]),
         "assembly_triple": (wrapper(assembly_triple[0]),
                             wrapper(assembly_triple[1], dtype=jnp.int64),
                             wrapper(assembly_triple[2], dtype=jnp.int64)),
         "hypervis_scaling": wrapper(hypervis_scaling),
         "triples_send": triples_send,
         "triples_receive": triples_recv
         }
  metdet = ret["metric_determinant"]
  # if use_wrapper and wrapper_type == "torch":
  #   from .config import torch
  #   ret["dss_matrix"] = torch.sparse_coo_tensor((dss_triple[2], dss_triple[3]),
  #                                                dss_triple[0],
  #                                                size=(NELEM * npt * npt, NELEM * npt * npt))
  if not wrapped:
    ret["vertex_redundancy"] = vert_red_local
    ret["vertex_redundancy_send"] = vert_red_send
    ret["vertex_redundancy_receive"] = vert_red_recv
  send_dims = {}
  for proc_idx in triples_send.keys():
    send_dims[str(proc_idx)] = triples_send[proc_idx][0].size
  grid_dims = frozendict(N=metdet.size, shape=metdet.shape, npt=npt, num_elem=metdet.shape[0], **send_dims)
  return ret, grid_dims


def get_grid_deformation_metrics(grid, npt):
  eigs, _ = jnp.linalg.eigh(grid["metric_inverse"])
  max_svd = jnp.sqrt(jnp.max(eigs, axis=-1))
  min_svd = jnp.sqrt(jnp.min(eigs, axis=-1))
  dx_short = 1.0 / (max_svd * 0.5 * (npt - 1))
  dx_long = 1.0 / (min_svd * 0.5 * (npt - 1))
  return max_svd, dx_short, dx_long


def get_global_grid_deformation_metrics(h_grid, dims):
  L2_jac_inv, dx_short, dx_long = get_grid_deformation_metrics(h_grid, dims["npt"])
  max_norm_jac_inv = global_max(jnp.max(L2_jac_inv))
  max_min_dx = global_max(jnp.max(dx_short))
  min_max_dx = global_min(jnp.min(dx_long))
  return max_norm_jac_inv, max_min_dx, min_max_dx


def get_cfl(h_grid, radius_earth, diffusion_config, dims, sphere=True):
  #
  # estimate various CFL limits
  # Credit: This is basically copy-pasted from CAM-SE/HOMME

  # Courtesy of Paul Ullrich, Jared Whitehead
  lambda_maxs = {3: 1.5,
                 4: 2.74,
                 5: 4.18,
                 6: 5.86,
                 7: 7.79,
                 8: 10.0}

  lambda_viss = {3: 12.0,
                 4: 30.0,
                 5: 91.6742,
                 6: 190.117,
                 7: 374.7788,
                 8: 652.3015}

  npt = dims["npt"]
  scale_inv = 1.0 / radius_earth if sphere else 1.0

  assert npt in lambda_maxs.keys() and npt in lambda_viss.keys(), "Stability characteristics not calculated for {npt}"
  lambda_max = lambda_maxs[npt]
  lambda_vis = lambda_viss[npt]
  minimum_gauss_weight = jnp.min(h_grid["gll_weights"])

  hypervis_scaling = h_grid["hypervis_scaling"]

  max_norm_jac_inv, max_min_dx, min_min_dx = get_global_grid_deformation_metrics(h_grid, dims)

  # tensorHV.  New eigenvalues are the eigenvalues of the tensor V
  # formulas here must match what is in cube_mod.F90
  # for tensorHV, we scale out the rearth dependency
  lam = max_norm_jac_inv**2

  norm_jac_inv_hvis_tensor = (lambda_vis**2) * (max_norm_jac_inv**4) * (lam**(-hypervis_scaling / 2.0))

  norm_jac_inv_hvis_const = (lambda_vis**2) * (1.0 / radius_earth * max_norm_jac_inv)**4
  if "tensor_hypervis" in diffusion_config.keys():
    norm_jac_inv_hvis = norm_jac_inv_hvis_tensor
  else:
    norm_jac_inv_hvis = norm_jac_inv_hvis_const

  nu_div_fact = 1.0 if "tensor_hypervis" in diffusion_config.keys() else diffusion_config["nu_div_factor"]
  rkssp_euler_stability = minimum_gauss_weight / (120.0 * max_norm_jac_inv * scale_inv)
  rk2_tracer = 1.0 / (120.0 * max_norm_jac_inv * lambda_max * scale_inv)
  gravit_wave_stability = 1.0 / (342.0 * max_norm_jac_inv * lambda_max * scale_inv)
  hypervis_stability_dpi = 1.0 / (diffusion_config["nu_dpi"] * norm_jac_inv_hvis)
  hypervis_stability_vort = 1.0 / (diffusion_config["nu"] * norm_jac_inv_hvis)
  hypervis_stability_div = 1.0 / (nu_div_fact * diffusion_config["nu"] * norm_jac_inv_hvis)
  return ({"dt_rkssp_euler": rkssp_euler_stability,
           "dt_rk2_tracer": rk2_tracer,
           "dt_gravity_wave": gravit_wave_stability,
           "dt_hypervis_scalar": hypervis_stability_dpi,
           "dt_hypervis_vort": hypervis_stability_vort,
           "dt_hypervis_div": hypervis_stability_div},
          {"max_norm_jac_inv": max_norm_jac_inv,
           "max_min_dx": max_min_dx,
           "min_min_dx": min_min_dx,
           "lambda_vis": lambda_vis,
           "scale_inv": scale_inv})


def postprocess_grid(grid, dims):
  npt = dims["npt"]
  spectral = init_spectral(npt)

  def project_matrix(matrix):
    upper_left = matrix[:, :, :, 0, 0]
    upper_right = matrix[:, :, :, 0, 1]
    lower_left = matrix[:, :, :, 1, 0]
    lower_right = matrix[:, :, :, 1, 1]
    upper_left_out = project_scalar_global([upper_left], grid, dims)[0]
    upper_right_out = project_scalar_global([upper_right], grid, dims)[0]
    lower_left_out = project_scalar_global([lower_left], grid, dims)[0]
    lower_right_out = project_scalar_global([lower_right], grid, dims)[0]
    left_col = jnp.stack((upper_left_out, lower_left_out), axis=-1)
    right_col = jnp.stack((upper_right_out, lower_right_out), axis=-1)
    return jnp.stack((left_col, right_col), axis=-1)

  tensor_cont = project_matrix(grid["viscosity_tensor"])
  tensor_bilinear = np.zeros_like(tensor_cont)
  for i_idx in range(npt):
    for j_idx in range(npt):
        beta = spectral["gll_points"][i_idx]
        alpha = spectral["gll_points"][j_idx]
        v0 = tensor_cont[:, 0, 0, :, :]
        v1 = tensor_cont[:, 0, npt - 1, :, :]
        v2 = tensor_cont[:, npt - 1, 0, :, :]
        v3 = tensor_cont[:, npt - 1, npt - 1, :, :]
        tensor_bilinear[:, i_idx, j_idx, :, :] = bilinear(v0,
                                                          v1,
                                                          v2,
                                                          v3, alpha, beta)

  grid["viscosity_tensor"] = device_wrapper(tensor_bilinear)
  return grid
