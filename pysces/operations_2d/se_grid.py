from ..config import np, device_wrapper, use_wrapper, device_unwrapper, jnp
from scipy.sparse import coo_array
from frozendict import frozendict
from ..distributed_memory.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from ..spectral import init_spectral
#from ..mesh_generation.mesh import vert_red_flat_to_hierarchy, vert_red_hierarchy_to_flat


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


# def triage_vert_redundancy(vert_redundancy_gll,
#                            proc_idx, decomp):
#   # current understanding: this works because the outer
#   # three for loops will iterate in exactly the same order for
#   # the sending and recieving processor
#   vert_red_flat = vert_red_hierarchy_to_flat(vert_redundancy_gll)
#   vert_red_local_flat, vert_red_send, vert_red_receive = triage_vert_redundancy_flat(vert_red_flat, proc_idx, decomp)
#   vert_red_local = vert_red_flat_to_hierarchy(vert_red_local_flat)

  return vert_red_local, vert_red_send, vert_red_receive

def init_hypervis_tensor_copypaste(met_inv, jacobian_inv, hypervis_scaling=3.2):

  # matricies for tensor hyper-viscosity
  # compute eigenvectors of metinv (probably same as computed above)
  # M = elem%metinv(i,j,:,:)

  eig_one = (met_inv[:, :, :, 0, 0] + met_inv[:, :, :, 1, 1] + np.sqrt(4.0*met_inv[:, :, :, 0, 1] * met_inv[:, :, :, 1,0] +
              (met_inv[:, :, :, 0, 0] - met_inv[:, :, :, 1, 1])**2))/2.0
  eig_two = (met_inv[:, :, :, 0, 0] + met_inv[:, :, :, 1, 1] - np.sqrt(4.0 * met_inv[:, :, :, 0, 1]*met_inv[:, :, :, 1, 0] +
              (met_inv[:, :, :, 0, 0] - met_inv[:, :, :, 1, 1])**2))/2.0
          
  # use DE to store M - Lambda, to compute eigenvectors
  char_mat =np.copy(met_inv)
  char_mat[:, :, :, 0, 0] = char_mat[:, :, :, 0, 0] - eig_one
  char_mat[:, :, :, 1, 1] = char_mat[:, :, :, 1, 1] - eig_one
  E = np.copy(met_inv)
  indices = np.argmax(np.abs(char_mat).reshape(*char_mat.shape[:3], 4))
  indices_2d = np.unravel_index(indices, shape=(2, 2))
  if np.max(np.abs(char_mat)) == 0:
    E[:, :, :, 0, 0] = 1.0
    E[:, :, :, 1, 0] = 0.0
  elif (indices_2d[:, :, :, 0] == 0 and indices_2d[:, :, :, 1] == 0):
    E[:, :, :, 1, 0] = 1.0
    E[:, :, :, 0, 0] = -char_mat[:, :, :, 1, 0] / char_mat[:, :, :, 0, 0]
  elif ( indices_2d[:, :, :, 0] == 0 and indices_2d[:, :, :, 1] == 1):
      E[:, :, :, 1, 0] = 1.0
      E[:, :, :, 0, 0] = -char_mat[:, :, :, 1, 1] / char_mat[:, :, :, 0, 1]
  elif ( indices_2d[:, :, :, 0] == 1 and indices_2d[:, :, :, 1] == 0):
      E[:, :, :, 0, 0] = 1.0
      E[:, :, :, 1, 0] = -char_mat[:, :, :, 0, 0] / char_mat[:, :, :, 1,0]
  elif ( indices_2d[:, :, :, 0] == 1 and indices_2d[:, :, :, 1] == 1):
      E[:, :, :, 0, 0] = 1.0
      E[:, :, :, 1, 0] = -char_mat[:, :, :, 0, 1] / char_mat[:, :, :, 1, 1]

  # the other eigenvector is orthgonal:
  E[:, :, :, 0, 1] = -E[:, :, :, 1,0]
  E[:, :, :, 1, 1]= E[:, :, :, 0, 0]

  #normalize columns
  for idx in range(2):
    E[:, :, :, :, idx] /= np.sqrt(np.sum(E[:, :, :, :, idx]*
                                         E[:, :, :, :, idx], axis=-1))[:, :, :, np.newaxis]; 

  # OBTAINING TENSOR FOR HV:

  # Instead of the traditional scalar Laplace operator \grad \cdot \grad
  # we introduce \grad \cdot V \grad
  # where V = D E LAM LAM^* E^T D^T. 
  # Recall (metric_tensor)^{-1}=(D^T D)^{-1} = E LAM E^T.
  # Here, LAM = diag( 4/((np-1)dx)^2 , 4/((np-1)dy)^2 ) = diag(  4/(dx_elem)^2, 4/(dy_elem)^2 )
  # Note that metric tensors and LAM correspondingly are quantities on a unit sphere.

  # This motivates us to use V = D E LAM LAM^* E^T D^T
  # where LAM^* = diag( nu1, nu2 ) where nu1, nu2 are HV coefficients scaled like (dx)^{hv_scaling/2}, (dy)^{hv_scaling/2}.
  # (Halves in powers come from the fact that HV consists of two Laplace iterations.)

  # Originally, we took LAM^* = diag(
  #  1/(eig(1)**(hypervis_scaling/4.0d0))*(rearth**(hypervis_scaling/2.0d0))
  #  1/(eig(2)**(hypervis_scaling/4.0d0))*(rearth**(hypervis_scaling/2.0d0)) ) = 
  #  = diag( lamStar1, lamStar2)
  #  \simeq ((np-1)*dx_sphere / 2 )^hv_scaling/2 = SQRT(OPERATOR_HV)
  # because 1/eig(...) \simeq (dx_on_unit_sphere)^2 .
  # Introducing the notation OPERATOR = lamStar^2 is useful for conversion formulas.

  # This leads to the following conversion formula: nu_const is nu used for traditional HV on uniform grids
  # nu_tensor = nu_const * OPERATOR_HV^{-1}, so
  # nu_tensor = nu_const *((np-1)*dx_sphere / 2 )^{ - hv_scaling} or
  # nu_tensor = nu_const *(2/( (np-1) * dx_sphere) )^{hv_scaling} .
  # dx_sphere = 2\pi *rearth/(np-1)/4/NE
  # [nu_tensor] = [meter]^{4-hp_scaling}/[sec]

  # (1) Later developments:
  # Apply tensor V only at the second Laplace iteration. Thus, LAM^* should be scaled as (dx)^{hv_scaling}, (dy)^{hv_scaling},
  # see this code below:
  #          DEL(1:2,1) = (lamStar1**2) *eig(1)*DE(1:2,1)
  #          DEL(1:2,2) = (lamStar2**2) *eig(2)*DE(1:2,2)

  # (2) Later developments:
  # Bringing [nu_tensor] to 1/[sec]:
  #	  lamStar1=1/(eig(1)**(hypervis_scaling/4.0d0)) *(rearth**2.0d0)
  #	  lamStar2=1/(eig(2)**(hypervis_scaling/4.0d0)) *(rearth**2.0d0)
  # OPERATOR_HV = ( (np-1)*dx_unif_sphere / 2 )^{hv_scaling} * rearth^4
  # Conversion formula:
  # nu_tensor = nu_const * OPERATOR_HV^{-1}, so
  # nu_tensor = nu_const *( 2*rearth /((np-1)*dx))^{hv_scaling} * rearth^{-4.0}.

  # For the baseline coefficient nu=1e15 for NE30, 
  # nu_tensor=7e-8 (BUT RUN TWICE AS SMALL VALUE FOR NOW) for hv_scaling=3.2
  # and 
  # nu_tensor=1.3e-6 for hv_scaling=4.0.

  #matrix D*E
  DE = np.einsum("fijgs,fijsh->fijgh", jacobian_inv, E)
  #DE(1,1)=sum(elem%D(i,j,1,:)*E(:,1))
  #DE(1,2)=sum(elem%D(i,j,1,:)*E(:,2))
  #DE(2,1)=sum(elem%D(i,j,2,:)*E(:,1))
  #DE(2,2)=sum(elem%D(i,j,2,:)*E(:,2))

  lamStar1 = 1.0 / (eig_one**(hypervis_scaling/4.0)) #* (rearth**2.0)
  lamStar2 = 1.0 / (eig_two**(hypervis_scaling/4.0)) #* (rearth**2.0)

  # matrix (DE) * Lam^* * Lam , tensor HV when V is applied at each Laplace calculation
  #          DEL(1:2,1) = lamStar1*eig(1)*DE(1:2,1)
  #          DEL(1:2,2) = lamStar2*eig(2)*DE(1:2,2)

  #matrix (DE) * (Lam^*)^2 * Lam, tensor HV when V is applied only once, at the last Laplace calculation
  #will only work with hyperviscosity, not viscosity
  DEL = np.zeros_like(DE)
  DEL[:, :, :, :, 0] = (lamStar1**2 * eig_one)[:, :, :, np.newaxis] * DE[:, :, :, :, 0]
  DEL[:, :, :, :, 1] = (lamStar2**2 * eig_two)[:, :, :, np.newaxis] * DE[:, :, :, :, 1]

  #matrix (DE) * Lam^* * Lam  *E^t *D^t or (DE) * (Lam^*)^2 * Lam  *E^t *D^t 
  viscosity_tensor = np.einsum("fijgs,fijsh->fijgh", DEL, DE)
  # V(1,1)=sum(DEL(1,:)*DE(1,:))
  # V(1,2)=sum(DEL(1,:)*DE(2,:))
  # V(2,1)=sum(DEL(2,:)*DE(1,:))
  # V(2,2)=sum(DEL(2,:)*DE(2,:))

  # NOTE: missing rearth**4 scaling compared to HOMME code
	# elem%tensorVisc(i,j,:,:)=V(:,:)
  return viscosity_tensor

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
  send_dims = {}
  for proc_idx in triples_send.keys():
    send_dims[str(proc_idx)] = triples_send[proc_idx][0].size
  grid_dims = frozendict(N=metdet.size, shape=metdet.shape, npt=npt, num_elem=metdet.shape[0], **send_dims)
  return ret, grid_dims
