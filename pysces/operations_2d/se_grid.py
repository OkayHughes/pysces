from ..config import np, device_wrapper, use_wrapper, device_unwrapper, jnp
from scipy.sparse import coo_array
from frozendict import frozendict
from .assembly import project_scalar
from ..mesh_generation.coordinate_utils import bilinear
from ..distributed_memory.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from ..distributed_memory.multiprocessing import project_scalar_triple_mpi, global_max, global_min
from ..spectral import init_spectral


def init_assembly_matrix(NELEM, npt, assembly_triple):
  data, rows, cols = assembly_triple
  assembly_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM * npt * npt))
  return assembly_matrix


def init_assembly_local(NELEM, npt, vert_redundancy_local):
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  # hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size).reshape(index_hack.shape)

  data = []
  rows = []
  cols = []

  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_local:
    data.append(1.0)
    rows.append(index_hack[remote_face_id, remote_i, remote_j])
    cols.append(index_hack[local_face_idx, local_i, local_j])

  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return (np.array(data, dtype=np.float64),
          np.array(rows, dtype=np.int64),
          np.array(cols, dtype=np.int64))


def init_assembly_global(NELEM, npt, vert_redundancy_send, vert_redundancy_receive):
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  # hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size, dtype=np.int64).reshape(index_hack.shape)

  # convention: when scaled=True, remote values are
  # pre-multiplied by numerator
  # divided by total mass matrix on receiving end
  triples_receive = {}
  triples_send = {}
  for vert_redundancy, triples, transpose in zip([vert_redundancy_receive, vert_redundancy_send],
                                                 [triples_receive, triples_send],
                                                 [False, True]):
    for source_proc_idx in vert_redundancy.keys():
      data = []
      rows = []
      cols = []
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy[source_proc_idx]):
        data.append(1.0)
        if transpose:
          cols.append(index_hack[target_local_idx, target_i, target_j])
          rows.append(col_idx)
        else:
          rows.append(index_hack[target_local_idx, target_i, target_j])
          cols.append(col_idx)
      triples[source_proc_idx] = (np.array(data, dtype=np.float64),
                                  np.array(rows, dtype=np.int64),
                                  np.array(cols, dtype=np.int64))
  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return triples_send, triples_receive


def triage_vert_redundancy_flat(vert_redundancy_gll_flat,
                                proc_idx, decomp):
  # current understanding: this works because the outer
  # three for loops will iterate in exactly the same order for
  # the sending and recieving processor
  vert_redundancy_local = []
  vert_redundancy_send = {}
  vert_redundancy_receive = {}

  for ((target_global_idx, target_i, target_j),
       (source_global_idx, source_i, source_j)) in vert_redundancy_gll_flat:
    target_proc_idx = elem_idx_global_to_proc_idx(target_global_idx, decomp)
    source_proc_idx = elem_idx_global_to_proc_idx(source_global_idx, decomp)
    if (target_proc_idx == proc_idx and source_proc_idx == proc_idx):
      target_local_idx = int(global_to_local(target_global_idx, proc_idx, decomp))
      source_local_idx = int(global_to_local(source_global_idx, proc_idx, decomp))
      vert_redundancy_local.append(((target_local_idx, target_i, target_j),
                                    (source_local_idx, source_i, source_j)))
    elif (target_proc_idx == proc_idx and not
          source_proc_idx == proc_idx):
      target_local_idx = int(global_to_local(target_global_idx, proc_idx, decomp))
      if source_proc_idx not in vert_redundancy_receive.keys():
        vert_redundancy_receive[source_proc_idx] = []
      vert_redundancy_receive[source_proc_idx].append(((target_local_idx, target_i, target_j)))
    elif (not target_proc_idx == proc_idx and
          source_proc_idx == proc_idx):
      source_local_idx = int(global_to_local(source_global_idx, proc_idx, decomp))
      if target_proc_idx not in vert_redundancy_send.keys():
        vert_redundancy_send[target_proc_idx] = []
      vert_redundancy_send[target_proc_idx].append(((source_local_idx, source_i, source_j)))
  return vert_redundancy_local, vert_redundancy_send, vert_redundancy_receive


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



def init_hypervis_tensor(met_inv, jacobian, hypervis_scaling=3.2):
  """
  Initialize the metric tensor used to encode anisotropic resolution-dependent hyperviscosity 
  for unstructured grids.

  Parameters
  ----------
  met_inv : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_idx, alpha_beta_idx], Float]`
      The inverse metric tensor jacobian_inv*transpose(jacobian_inv)
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  jacobian : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_idx, alpha_beta_idx], Float]`
      The jacobian matrix that takes covariant/contravariant vectors on the 
      reference element to spherical coordinates.
  hypervis_scaling: `Float`, default=3.2
      Power to use in resolution-dependent hyperviscosity 
  Notes
  -----
  * We assume here that tensor hyperviscosity is applied as 
  ∇·(V ∇(∆^{2(lap_ord-1)} f). For example, for default fourth-order hyperviscosity 
  `lap_ord = 2`, hyperviscosity is calculated as ∇·V ∇A[∇·∇f], where A is the
  spectral element projection operator, e.g., assembly.project_scalar.
  * The viscosity tensor can be understood in the following way. We will focus on 
  fourth order hyperviscosity for simplicity. Recall that the point of hyperviscosity
  is to heavily damp unresolvable grid-scale flow features
  without artificially damping resolved flow.
  On a quasi-uniform grid, hyperviscosity can be applied as ν_const ∆^2 f (note that 
  this laplacian is applied on the unit sphere). Empirical experiments show that
  ν_const = 1e15(ne30/neXX)^(3.2) is a good
  choice of a spatio-temporally homogeneous hyperdiffusion constant.
  Using non-uniform grids introduces two complications. Firstly,
  some grid cells should have much smaller area than others,
  because this is the point of using variable-resolution modeling. 
  This might motivate using a gridpoint-dependent ν, such as 
  calculating the ratio of the area of the element that contains it to
  the average grid cell area of an NE30 grid. However, grid cells can also
  be quite distorted (especially in regions where the grid is transitioning from
  a coarser to a finer grid). In a grid cell that is twice as tall as it is wide
  (imagine we are woring on an x, y tangent plane)
  a 4∆y feature may be well resolved in the x direction, but poorly resolved due to the
  increased distance between quadrature points in the y direction. The 
  standard spherical Laplacian ∇·∇f is constructed so that a grid with distorted elements
  will still converge to correct solution as grid resolution increases. This motivates the
  construction of a metric tensor V (recall: this is a positive definite matrix
  that induces a modified notion of, e.g., the length of a vector) that allows us to 
  damp small-scale features in a way that re-introduces grid distortion.
  * To do this, we do an eigendomposition of the the (symmetric) inverse metric M= J^{-1}J^{-T} 
  into M = E Λ E^T (recall, E are unit eigenvectors and M is symmetric, so E, E^T are inverses of each other),
  where Λ = diag((4/((npt-1)∆x)^2, 4/((npt-1)∆y)^2). Note that 4 is the area of the reference element [-1, 1]^2.
  Because most unstructured grids bilinearly map the reference element to the sphere 
  in a way that is approximately bilinear on > nx=15 grids, ∆x and ∆y are essentially the cartesian length between grid points,
  and (npt-1)∆x, (npt-1)∆y are ∆x_elem, ∆y_elem, respectively. We then define 
  Λ^* = diag(1/λ_0^(-hv_scaling/2.0), 1/λ_1^(-hv_scaling/2.0)), and it turns out that
  V = J E Λ Λ^* E^T J^T satisfies the properties we're looking for. In practice,
  the result of hyperviscosity must be scaled by radius_earth**4. Once that scaling is done,
  one can derive an equivalent ν_tensor given a ν_const, using the grid-dependent 
  value h = ((np-1)*∆x/2)^{hv_scaling} * radius_earth**4, and find 
  ν_tensor = ν_const / h. Therefore, a spatially uniform hyperviscosity with ν_const=10^15
  would have ν_tensor=7e-8 (though in practice most runs use 3.4e-8).
  [TODO] Make this explanation more accessible

  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.

  Returns
  -------
  viscosity_tensor: `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      Anisotropy tensor applied in last application of laplacian within hyperviscosity.
  """

  eigs, evecs_normed = jnp.linalg.eigh(met_inv)
  lam_star = 1.0 / (eigs**(hypervis_scaling / 4.0))
  met_inv_scaled = jnp.einsum("fijmc, fijnc, fijc, fijc->fijmn", evecs_normed, evecs_normed, eigs, lam_star**2)

  # NOTE: missing rearth**4 scaling compared to HOMME code
  viscosity_tensor = jnp.einsum("fijmn, fijsm, fijrn -> fijsr",
                                met_inv_scaled,
                                jacobian,
                                jacobian)
  return viscosity_tensor, hypervis_scaling


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
         "physical_coords_to_cartesian": subset_wrapper(physical_coords_to_cartesian),
         "jacobian": subset_wrapper(gll_to_sphere_jacobian),
         "jacobian_inv": subset_wrapper(gll_to_sphere_jacobian_inv),
         "recip_met_det": subset_wrapper(rmetdet),
         "met_det": subset_wrapper(metdet),
         "mass_mat": subset_wrapper(mass_mat),
         "mass_matrix_inv": subset_wrapper(inv_mass_mat),
         "met_inv": subset_wrapper(met_inv),
         "mass_matrix": subset_wrapper(mass_matrix),
         "viscosity_tensor": subset_wrapper(viscosity_tensor),
         "deriv": wrapper(spectrals["deriv"]),
         "gll_weights": wrapper(spectrals["gll_weights"]),
         "assembly_triple": (wrapper(assembly_triple[0]),
                             wrapper(assembly_triple[1], dtype=jnp.int64),
                             wrapper(assembly_triple[2], dtype=jnp.int64)),
         "hypervis_scaling": wrapper(hypervis_scaling),
         "triples_send": triples_send,
         "triples_receive": triples_recv
         }
  metdet = ret["met_det"]
  # if use_wrapper and wrapper_type == "torch":
  #   from .config import torch
  #   ret["dss_matrix"] = torch.sparse_coo_tensor((dss_triple[2], dss_triple[3]),
  #                                                dss_triple[0],
  #                                                size=(NELEM * npt * npt, NELEM * npt * npt))
  if not wrapped:
    ret["vert_redundancy"] = vert_red_local
    ret["vert_redundancy_send"] = vert_red_send
    ret["vert_redundancy_receive"] = vert_red_recv
  send_dims = {}
  for proc_idx in triples_send.keys():
    send_dims[str(proc_idx)] = triples_send[proc_idx][0].size
  grid_dims = frozendict(N=metdet.size, shape=metdet.shape, npt=npt, num_elem=metdet.shape[0], **send_dims)
  return ret, grid_dims

def postprocess_grid(grid, dims, distributed=False):
  npt = dims["npt"]
  spectral = init_spectral(npt)
  def project_matrix(matrix):
    upper_left = np.copy(matrix[:, :, :, 0, 0])
    upper_right = np.copy(matrix[:, :, :, 0, 1])
    lower_left = np.copy(matrix[:, :, :, 1, 0])
    lower_right = np.copy(matrix[:, :, :, 1, 1])
    upper_left = project_scalar(upper_left, grid, dims)
    upper_right = project_scalar(upper_right, grid, dims)
    lower_left = project_scalar(lower_left, grid, dims)
    lower_right = project_scalar(lower_right, grid, dims)
    left_col = np.stack((upper_left, lower_left), axis=-1)
    right_col = np.stack((upper_right, lower_right), axis=-1)
    return np.stack((left_col, right_col), axis=-1)
  tensor_cont = project_matrix(grid["viscosity_tensor"])
  tensor_bilinear = np.zeros_like(tensor_cont)
  for i_idx in range(npt):
    for j_idx in range(npt):
        beta = spectral["gll_points"][i_idx]
        alpha = spectral["gll_points"][j_idx]
        v0 = tensor_cont[:, 0, 0, :, :]
        v1 = tensor_cont[:, 0, npt-1, :, :]
        v2 = tensor_cont[:, npt-1, 0, :, :]
        v3 = tensor_cont[:, npt-1, npt-1, :, :]
        tensor_bilinear[:, i_idx, j_idx, :, :] = bilinear(v0,
                                                          v1,
                                                          v2,
                                                          v3, alpha, beta)

  grid["viscosity_tensor"] = tensor_bilinear
  return grid


def get_grid_deformation_metrics(grid, npt):
  eigs, _ = jnp.linalg.eigh(grid["met_inv"])
  max_svd = jnp.sqrt(jnp.max(eigs, axis=-1))
  min_svd =  jnp.sqrt(jnp.min(eigs, axis=-1))
  dx_short = 1.0 / (max_svd*0.5*(npt-1))
  dx_long  = 1.0 / (min_svd*0.5*(npt-1))
  return max_svd, dx_short, dx_long