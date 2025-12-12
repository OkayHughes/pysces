from .config import np, npt, jax_wrapper, use_wrapper, wrapper_type
from .spectral import deriv
from scipy.sparse import coo_array
from frozendict import frozendict


def init_dss_matrix(metdet, inv_mass_mat, vert_redundancy_gll):

  NELEM = metdet.shape[0]
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  # hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size).reshape(index_hack.shape)

  data = []
  rows = []
  cols = []
  data_un = []

  for face_idx in range(NELEM):
    for i_idx in range(npt):
      for j_idx in range(npt):
        data.append(metdet[face_idx, i_idx, j_idx] * ((deriv["gll_weights"][i_idx] * deriv["gll_weights"][j_idx]) *
                                                      inv_mass_mat[face_idx, i_idx, j_idx]))
        data_un.append(inv_mass_mat[face_idx, i_idx, j_idx])
        rows.append(index_hack[face_idx, i_idx, j_idx])
        cols.append(index_hack[face_idx, i_idx, j_idx])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        data.append(metdet[local_face_idx, local_i, local_j] * ((deriv["gll_weights"][local_i] *
                                                                 deriv["gll_weights"][local_j]) *
                                                                inv_mass_mat[local_face_idx, local_i, local_j]))
        data_un.append(inv_mass_mat[local_face_idx, local_i, local_j])
        rows.append(index_hack[remote_face_id, remote_i, remote_j])
        cols.append(index_hack[local_face_idx, local_i, local_j])
  # sparse matrix representation makes eventual autograd port significantly easier
  dss_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM * npt * npt))
  dss_matrix_unscaled = coo_array((data_un, (rows, cols)), shape=(NELEM * npt * npt, NELEM * npt * npt))

  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return dss_matrix, dss_matrix_unscaled, (data, data_un, rows, cols)


def create_spectral_element_grid(latlon,
                                 gll_to_sphere_jacobian,
                                 gll_to_sphere_jacobian_inv,
                                 rmetdet,
                                 metdet,
                                 mass_mat,
                                 inv_mass_mat,
                                 vert_redundancy,
                                 jax=use_wrapper,
                                 device=""):
  dss_matrix, dss_matrix_unscaled, dss_triple = init_dss_matrix(metdet, inv_mass_mat, vert_redundancy)
  # note: test code sometimes sets jax=False to test jax vs stock numpy
  # this extra conditional is not extraneous.
  if jax:
    wrapper = jax_wrapper
  else:
    def wrapper(x):
      return x
  NELEM = metdet.shape[0]
  ret = {"physical_coords": wrapper(latlon),
         "jacobian": wrapper(gll_to_sphere_jacobian),
         "jacobian_inv": wrapper(gll_to_sphere_jacobian_inv),
         "recip_met_det": wrapper(rmetdet),
         "met_det": wrapper(metdet),
         "mass_mat": wrapper(mass_mat),
         "mass_matrix_inv": wrapper(inv_mass_mat),
         "met_inv": wrapper(np.einsum("fijgs, fijhs->fijgh",
                                      gll_to_sphere_jacobian_inv,
                                      gll_to_sphere_jacobian_inv)),
         "deriv": wrapper(deriv["deriv"]),
         "gll_weights": wrapper(deriv["gll_weights"]),
         "dss_triple": (wrapper(dss_triple[0]),
                        wrapper(dss_triple[1]),
                        wrapper(dss_triple[2]),
                        wrapper(dss_triple[3])),
         }

  if not jax:
    ret["vert_redundancy"] = vert_redundancy
    ret["dss_matrix"] = dss_matrix
    ret["dss_matrix_unscaled"] = dss_matrix_unscaled
  if use_wrapper and wrapper_type=="torch":
    from .config import torch
    ret["dss_matrix"] = torch.sparse_coo_tensor((dss_triple[2], dss_triple[3]), dss_triple[0], size=(NELEM * npt * npt, NELEM * npt * npt))
    ret["dss_matrix_unscaled"] = torch.sparse_coo_tensor((dss_triple[2], dss_triple[3]), dss_triple[1], size=(NELEM * npt * npt, NELEM * npt * npt))
  grid_dims = frozendict(N=metdet.size, shape=metdet.shape, npt=npt, num_elem=metdet.shape[0])
  return ret, grid_dims
