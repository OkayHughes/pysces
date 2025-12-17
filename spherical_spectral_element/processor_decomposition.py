from .config import np
from .grid_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE


def hilbert_curve(n_subdiv):
  A,B,C,D = 0, 1, 2, 3
  arr_prev = C *  np.ones((1,1), dtype=np.int64)
  idxs_prev = np.zeros((1,1), dtype=np.int64)
  for iter in range(n_subdiv):
    idxs_next = np.zeros((2**(iter+1), 2**(iter+1)), dtype=np.int64)
    arr_next = np.zeros((2**(iter+1), 2**(iter+1)), dtype=np.int64)
    for i_idx in range(int(2**iter)):
      for j_idx in range(int(2**iter)):
        prev_ind = 4 * idxs_prev[i_idx, j_idx]
        if arr_prev[i_idx, j_idx] == A:
          arr_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[A, A],[D, B]]
          idxs_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[prev_ind + 1, prev_ind + 2],[prev_ind + 0,prev_ind + 3]]
        if arr_prev[i_idx, j_idx] == B:
          arr_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[B, C],[B, A]]
          idxs_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[prev_ind + 1, prev_ind + 0],[prev_ind + 2,prev_ind + 3]]
        if arr_prev[i_idx, j_idx] == C:
          arr_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[D, B],[C, C]]
          idxs_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[prev_ind + 3, prev_ind + 0],[prev_ind + 2,prev_ind + 1]]
        if arr_prev[i_idx, j_idx] == D:
          arr_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[C, D],[A, D]]
          idxs_next[2*i_idx:2*(i_idx+1), 2*j_idx:2*(j_idx+1)] = [[prev_ind + 3, prev_ind + 2],[prev_ind + 0,prev_ind + 1]]
    idxs_prev, idxs_next = idxs_next, idxs_prev
    arr_prev, arr_next = arr_next, arr_prev
  return idxs_prev


def get_face_idx_pos(lat, lon):
  # assumes lon \in [0, 2*np.pi]
  lat[lat > np.pi/2.0 - 1e-4] = np.pi/2.0 - 1e-4
  lat[lat < -np.pi/2.0 + 1e-4] = -np.pi/2.0 + 1e-4

  equator_panel_centers = np.arange(0, 2*np.pi-1e-8, np.pi/2) + np.pi/4
  lon_diff = np.abs(lon.reshape((*lon.shape, 1)) - equator_panel_centers.reshape((*np.ones_like(lon.shape), equator_panel_centers.size)))
  face_idx = np.argmin(lon_diff, axis=-1)
  lon_loc = lon - equator_panel_centers[face_idx.flatten()].reshape(face_idx.shape)
  y_provisional = np.tan(lat) / np.cos(lon_loc)
  x_provisional = np.tan(lon_loc)
  top_mask = y_provisional > 1.0
  bottom_mask = y_provisional < -1.0
  front_mask = face_idx == 0
  left_mask = face_idx == 3
  right_mask = face_idx == 2
  back_mask = face_idx == 1


  y_provisional[top_mask] = -np.cos(lon[top_mask]+np.pi/4) / np.tan(lat[top_mask])
  y_provisional[bottom_mask] = -np.cos(lon[bottom_mask]+np.pi/4) / np.tan(lat[bottom_mask])
  x_provisional[top_mask] = np.sin(lon[top_mask]+np.pi/4) / np.tan(lat[top_mask])
  x_provisional[bottom_mask] = -np.sin(lon[bottom_mask] + np.pi/4) / np.tan(lat[bottom_mask])
  face_idx[left_mask] = LEFT_FACE
  face_idx[right_mask] = RIGHT_FACE
  face_idx[front_mask] = FRONT_FACE
  face_idx[back_mask] = BACK_FACE
  face_idx[top_mask] = TOP_FACE
  face_idx[bottom_mask] = BOTTOM_FACE
  return face_idx, x_provisional, y_provisional


def processor_id_to_range(proc_idx, num_faces, num_procs):
  # probably not good for extremely high numbers of
  # MPI ranks.

  stride = int(np.floor(num_faces/num_procs))
  beginning_idx = proc_idx * stride
  if proc_idx == num_procs - 1:
    end_idx = num_faces
  else:
    end_idx = (proc_idx + 1) * stride
  return beginning_idx, end_idx


def get_decomp(num_faces, num_procs):
  segments = []
  for proc_idx in range(num_procs):
    segments.append(processor_id_to_range(proc_idx, num_faces, num_procs))
  return segments


def create_mapping(n_subdiv, latlons):
  face_idx, x, y = get_face_idx_pos(latlons[:, 0], latlons[:, 1])
  idxs = hilbert_curve(n_subdiv)
  grid_pos_1d = np.linspace(-1.0, 1.0, 2**n_subdiv)
  id_x = np.argmin(np.abs(x[np.newaxis, :] - grid_pos_1d[:, np.newaxis]), axis=0)
  id_y = np.argmin(np.abs(y[np.newaxis, :] - grid_pos_1d[:, np.newaxis]), axis=0)
  idxs_coarse = idxs[id_x, id_y]
  idxs_coarse += face_idx * idxs.size
  index_map = np.argsort(idxs_coarse)
  return index_map


def local_to_global(elem_idxs_local, proc_idx, decomp):
  return elem_idxs_local + decomp[proc_idx][0]


def global_to_local(elem_idxs_global, proc_idx, decomp):
  return elem_idxs_global - decomp[proc_idx][0]


def elem_idx_global_to_proc_idx(elem_idxs_global, decomp):
  out = np.zeros_like(elem_idxs_global, dtype=np.int64)
  for proc_idx, (begin, end) in enumerate(decomp):
    mask = np.logical_and(elem_idxs_global < end, elem_idxs_global >= begin)
    out[mask] = proc_idx
  return out
