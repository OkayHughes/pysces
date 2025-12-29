from .config import np, DEBUG
from .mathops import bilinear, bilinear_jacobian
from .grid_definitions import TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE
from .spectral import init_spectral


def mesh_to_cart_bilinear(face_position, npt):

  spectrals = init_spectral(npt)
  cart_dim = face_position.shape[2]
  NFACES = face_position.shape[0]

  gll_position = np.zeros(shape=(NFACES, npt, npt, cart_dim))
  gll_jacobian = np.zeros(shape=(NFACES, npt, npt, cart_dim, 2))

  for i_idx in range(npt):
    for j_idx in range(npt):
        alpha = spectrals["gll_points"][i_idx]
        beta = spectrals["gll_points"][j_idx]
        gll_position[:, i_idx, j_idx, :] = bilinear(face_position[:, 0, :],
                                                    face_position[:, 1, :],
                                                    face_position[:, 2, :],
                                                    face_position[:, 3, :], alpha, beta)

        dphys_dalpha, dphys_dbeta = bilinear_jacobian(face_position[:, 0, :],
                                                      face_position[:, 1, :],
                                                      face_position[:, 2, :],
                                                      face_position[:, 3, :], alpha, beta)
        gll_jacobian[:, i_idx, j_idx, :, 0] = dphys_dalpha
        gll_jacobian[:, i_idx, j_idx, :, 1] = dphys_dbeta

  return gll_position, gll_jacobian


def gen_gll_redundancy(face_connectivity, vert_redundancy, npt):
  # temporary note: we can assume here that this is mpi-local.
  # note:
  # count DOFs
  vert_redundancy_gll = {}

  def wrap(elem_idx, i_idx, j_idx):
    if elem_idx not in vert_redundancy_gll.keys():
      vert_redundancy_gll[elem_idx] = dict()
    if (i_idx, j_idx) not in vert_redundancy_gll[elem_idx].keys():
      vert_redundancy_gll[elem_idx][(i_idx, j_idx)] = set()

  correct_orientation = set([(0, 1), (0, 2), (2, 3), (1, 3)])

  def is_forwards(v0, v1):
    return (v0, v1) in correct_orientation

  def vert_to_i_j(vert_idx):
    if vert_idx == 0:
      return 0, 0
    elif vert_idx == 1:
      return npt - 1, 0
    elif vert_idx == 2:
      return 0, npt - 1
    elif vert_idx == 3:
      return npt - 1, npt - 1

  def infer_edge(elem_adj_loc, edge_idx, free_idx):
    if edge_idx == TOP_EDGE:
      idx0, idx1 = (0, 1)
    elif edge_idx == LEFT_EDGE:
      idx0, idx1 = (0, 2)
    elif edge_idx == RIGHT_EDGE:
      idx0, idx1 = (1, 3)
    elif edge_idx == BOTTOM_EDGE:
      idx0, idx1 = (2, 3)

    # find only element that overlaps both vertices on edge
    elems = [x[0] for x in elem_adj_loc[idx0]]
    elem_id = list(filter(lambda x: x[0] in elems, elem_adj_loc[idx1]))
    if DEBUG:
      assert (len(elem_id) == 1)
    elem_idx_pair = elem_id[0][0]
    # determine which vertices element is paired to
    v0 = list(filter(lambda x: x[0] == elem_idx_pair, elem_adj_loc[idx0]))
    v1 = list(filter(lambda x: x[0] == elem_idx_pair, elem_adj_loc[idx1]))
    if DEBUG:
      assert (len(v0) == 1)
      assert (len(v1) == 1)
    v0 = v0[0][1]
    v1 = v1[0][1]
    v0_i_idx, v0_j_idx = vert_to_i_j(v0)
    v1_i_idx, v1_j_idx = vert_to_i_j(v1)

    # calculate i, j indices on paired element,
    # accounting for edge direction.
    # Note: switch statement above orients local edge
    # in the forward direction, so orientation of paired
    # edge can be inferred from the ordered pair (v0, v1)
    if v0_i_idx == v1_i_idx:
      i_idx_pair = v0_i_idx
      if is_forwards(v0, v1):
        j_idx_pair = free_idx
      else:
        j_idx_pair = npt - 1 - free_idx
    elif v0_j_idx == v1_j_idx:
      j_idx_pair = v0_j_idx
      if is_forwards(v0, v1):
        i_idx_pair = free_idx
      else:
        i_idx_pair = npt - free_idx - 1
    else:
      raise ValueError("vertex-edge pairing is scuffed")

    return elem_idx_pair, i_idx_pair, j_idx_pair

  # Note: conforming grids should have no singleton vertices of elements.
  for elem_idx in vert_redundancy.keys():
    for i_idx in range(npt):
      for j_idx in range(npt):
        corner_idx = -1
        if i_idx == 0 and j_idx == 0:
          corner_idx = 0
        elif i_idx == npt - 1 and j_idx == npt - 1:
          corner_idx = 3
        elif i_idx == 0 and j_idx == npt - 1:
          corner_idx = 2
        elif i_idx == npt - 1 and j_idx == 0:
          corner_idx = 1

        if corner_idx != -1:
          wrap(elem_idx, i_idx, j_idx)
          for elem_idx_pair, vert_idx_pair in vert_redundancy[elem_idx][corner_idx]:
            i_idx_pair, j_idx_pair = vert_to_i_j(vert_idx_pair)
            vert_redundancy_gll[elem_idx][(i_idx, j_idx)].add((elem_idx_pair, i_idx_pair, j_idx_pair))

        edge_idx = -1
        if j_idx != 0 and j_idx != npt - 1:
          if i_idx == 0:
            edge_idx = LEFT_EDGE
            free_idx = j_idx
          elif i_idx == npt - 1:
            edge_idx = RIGHT_EDGE
            free_idx = j_idx
        if i_idx != 0 and i_idx != npt - 1:
          if j_idx == 0:
            edge_idx = TOP_EDGE
            free_idx = i_idx
          elif j_idx == npt - 1:
            edge_idx = BOTTOM_EDGE
            free_idx = i_idx
        # Note 1: some duplicate work done here, but
        # all grid building code is run at most once!
        # Optimization is not important.
        # Note 2: gll points lying on an edge share at most one neighbor,
        # since we're working in 2D
        if edge_idx != -1:
            elem_idx_pair, i_idx_pair, j_idx_pair = infer_edge(vert_redundancy[elem_idx], edge_idx, free_idx)
            wrap(elem_idx, i_idx, j_idx)
            vert_redundancy_gll[elem_idx][(i_idx, j_idx)].add((elem_idx_pair, i_idx_pair, j_idx_pair))
  return vert_redundancy_gll
