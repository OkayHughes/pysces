from ..config import np, device_wrapper, use_wrapper, device_unwrapper, jnp
from scipy.sparse import coo_array
from frozendict import frozendict
from ..distributed_memory.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from ..spectral import init_spectral
from ..mesh_generation.mesh import vert_red_flat_to_hierarchy, vert_red_hierarchy_to_flat


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


def triage_vert_redundancy(vert_redundancy_gll,
                           proc_idx, decomp):
  # current understanding: this works because the outer
  # three for loops will iterate in exactly the same order for
  # the sending and recieving processor
  vert_red_flat = vert_red_hierarchy_to_flat(vert_redundancy_gll)
  vert_red_local_flat, vert_red_send, vert_red_receive = triage_vert_redundancy_flat(vert_red_flat, proc_idx, decomp)
  vert_red_local = vert_red_flat_to_hierarchy(vert_red_local_flat)

  return vert_red_local, vert_red_send, vert_red_receive


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


def create_spectral_element_grid(latlon,
                                 gll_to_sphere_jacobian,
                                 gll_to_sphere_jacobian_inv,
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
  ret = {"physical_coords": subset_wrapper(latlon),
         "jacobian": subset_wrapper(gll_to_sphere_jacobian),
         "jacobian_inv": subset_wrapper(gll_to_sphere_jacobian_inv),
         "recip_met_det": subset_wrapper(rmetdet),
         "met_det": subset_wrapper(metdet),
         "mass_mat": subset_wrapper(mass_mat),
         "mass_matrix_inv": subset_wrapper(inv_mass_mat),
         "met_inv": subset_wrapper(met_inv),
         "mass_matrix": subset_wrapper(mass_matrix),
         "deriv": wrapper(spectrals["deriv"]),
         "gll_weights": wrapper(spectrals["gll_weights"]),
         "assembly_triple": (wrapper(assembly_triple[0]),
                             wrapper(assembly_triple[1], dtype=jnp.int64),
                             wrapper(assembly_triple[2], dtype=jnp.int64)),
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

  ret["assembly_triple"] = assembly_triple
  ret["triples_recv"] = triples_send
  ret["triples_recv"] = triples_recv
  send_dims = {}
  for proc_idx in triples_send.keys():
    send_dims[str(proc_idx)] = triples_send[proc_idx][0].size
  grid_dims = frozendict(N=metdet.size, shape=metdet.shape, npt=npt, num_elem=metdet.shape[0], **send_dims)
  return ret, grid_dims
