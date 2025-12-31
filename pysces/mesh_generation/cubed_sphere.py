from ..config import np
from .mesh_definitions import BACKWARDS, FORWARDS
from .mesh_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE
from .mesh_definitions import TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE


face_topo = {TOP_FACE: {TOP_EDGE: (BACK_FACE, TOP_EDGE, BACKWARDS),
                        LEFT_EDGE: (LEFT_FACE, TOP_EDGE, FORWARDS),
                        RIGHT_EDGE: (RIGHT_FACE, TOP_EDGE, BACKWARDS),
                        BOTTOM_EDGE: (FRONT_FACE, TOP_EDGE, FORWARDS)},
             BOTTOM_FACE: {TOP_EDGE: (FRONT_FACE, BOTTOM_EDGE, FORWARDS),
                           LEFT_EDGE: (LEFT_FACE, BOTTOM_EDGE, BACKWARDS),
                           RIGHT_EDGE: (RIGHT_FACE, BOTTOM_EDGE, FORWARDS),
                           BOTTOM_EDGE: (BACK_FACE, BOTTOM_EDGE, BACKWARDS)},
             FRONT_FACE: {TOP_EDGE: (TOP_FACE, BOTTOM_EDGE, FORWARDS),
                          LEFT_EDGE: (LEFT_FACE, RIGHT_EDGE, FORWARDS),
                          RIGHT_EDGE: (RIGHT_FACE, LEFT_EDGE, FORWARDS),
                          BOTTOM_EDGE: (BOTTOM_FACE, TOP_EDGE, FORWARDS)},
             BACK_FACE: {TOP_EDGE: (TOP_FACE, TOP_EDGE, BACKWARDS),
                         LEFT_EDGE: (RIGHT_FACE, RIGHT_EDGE, FORWARDS),
                         RIGHT_EDGE: (LEFT_FACE, LEFT_EDGE, FORWARDS),
                         BOTTOM_EDGE: (BOTTOM_FACE, BOTTOM_EDGE, BACKWARDS)},
             LEFT_FACE: {TOP_EDGE: (TOP_FACE, LEFT_EDGE, FORWARDS),
                         LEFT_EDGE: (BACK_FACE, RIGHT_EDGE, FORWARDS),
                         RIGHT_EDGE: (FRONT_FACE, LEFT_EDGE, FORWARDS),
                         BOTTOM_EDGE: (BOTTOM_FACE, LEFT_EDGE, BACKWARDS)},
             RIGHT_FACE: {TOP_EDGE: (TOP_FACE, RIGHT_EDGE, BACKWARDS),
                          LEFT_EDGE: (FRONT_FACE, RIGHT_EDGE, FORWARDS),
                          RIGHT_EDGE: (BACK_FACE, LEFT_EDGE, FORWARDS),
                          BOTTOM_EDGE: (BOTTOM_FACE, RIGHT_EDGE, FORWARDS)}}
verts = [[-1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [-1.0, -1.0, 1.0],
         [1.0, -1.0, 1.0],
         [-1.0, 1.0, -1.0],
         [1.0, 1.0, -1.0],
         [-1.0, -1.0, -1.0],
         [1.0, -1.0, -1.0]]
vert_info = {TOP_FACE: np.array([verts[2], verts[3], verts[0], verts[1]]),
             BOTTOM_FACE: np.array([verts[4], verts[5], verts[6], verts[7]]),
             FRONT_FACE: np.array([verts[0], verts[1], verts[4], verts[5]]),
             BACK_FACE: np.array([verts[3], verts[2], verts[7], verts[6]]),
             LEFT_FACE: np.array([verts[2], verts[0], verts[6], verts[4]]),
             RIGHT_FACE: np.array([verts[1], verts[3], verts[5], verts[7]])}
axis_info = {TOP_FACE: (0, 1.0, 1, -1.0),
             BOTTOM_FACE: (0, 1.0, 1, 1.0),
             FRONT_FACE: (0, 1.0, 2, 1.0),
             BACK_FACE: (0, -1.0, 2, 1.0),
             LEFT_FACE: (1, 1.0, 2, 1.0),
             RIGHT_FACE: (1, -1.0, 2, 1.0)}

MAX_VERT_DEGREE = 4


def edge_to_vert(edge_id, is_forwards=FORWARDS):
  """
  Map an edge id of oriented vertex ids of a given element edge.

  Parameters
  ----------
  edge_id : int
      Index of edge within an element
  is_forwards: int, default=FORWARDS
      Is the edge reversed from its
      default orientation.

  Returns
  -------
  tuple[int, int]
      (vert_idx_0, vert_idx_1)
  
  Notes
  --------
  See mesh_definitions for grid conventions on
  vertex_idx, edge enumeration, and default direction.
  """
  if edge_id == TOP_EDGE:
    v_idx_in_0 = 0
    v_idx_in_1 = 1
  elif edge_id == LEFT_EDGE:
    v_idx_in_0 = 0
    v_idx_in_1 = 2
  elif edge_id == RIGHT_EDGE:
    v_idx_in_0 = 1
    v_idx_in_1 = 3
  elif edge_id == BOTTOM_EDGE:
    v_idx_in_0 = 2
    v_idx_in_1 = 3
  if is_forwards != FORWARDS:
    return v_idx_in_1, v_idx_in_0
  else:
    return v_idx_in_0, v_idx_in_1


def edge_match(nx, free_idx, id_edge_out, is_forwards):
  """
  Return the horizontal and vertical element indexes
  of the pair of an element across a cubed-sphere edge.

  Parameters
  ----------
  nx : int
      The number of elements on an edge of
      a cubed sphere face
  free_idx: int
      Whichever of the horizontal or vertical indices
      is varying across the current cubed-sphere edge.
  id_edge_out: int
      The the cubed-sphere edge of the paired face 
      along which edges are joined
  is_forwards:
      Are the two paired cubed-sphere edges
      oriented the same way?

  Returns
  -------
  tuple[int, int]
      (horizontal, vertical) element index of the
      paired element within the paired cubed-sphere face.
  """
  free_idx_flip = free_idx if is_forwards == FORWARDS else nx - free_idx - 1
  if id_edge_out == BOTTOM_EDGE:
    y_idx_out = nx - 1
    x_idx_out = free_idx_flip
  elif id_edge_out == TOP_EDGE:
    y_idx_out = 0
    x_idx_out = free_idx_flip
  elif id_edge_out == LEFT_EDGE:
    x_idx_out = 0
    y_idx_out = free_idx_flip
  elif id_edge_out == RIGHT_EDGE:
    x_idx_out = nx - 1
    y_idx_out = free_idx_flip
  return x_idx_out, y_idx_out


def elem_id_fn(nx, face_idx, x_idx, y_idx):
  """
  Maps an element within a
  regular grid on a cubed-sphere face 
  to a scalar index.

  Parameters
  ----------
  nx : int
      The number of elements on an edge of a cubed-sphere face.
  face_idx : int
      The index of the cubed-sphere face the element is located on.
  x_idx:
      the horizontal index of the element within the face's regular grid.
  y_idx:
      the vertical index of the element within the face's regular grid
      the 3rd param, by default 'value'

  Returns
  -------
  int
      The global index of the element.
  
  Notes
  -----
  This should be the inverse of inv_elem_id_fn
  """
  return face_idx * nx**2 + x_idx * nx + y_idx


def inv_elem_id_fn(nx, idx):
  """
  Map a global element index to the index of the
  cubed-sphere face it is located on, and its
  horizontal and vertical position in a regular grid.

  Parameters
  ----------
  nx : int
      The number of elements on a cubed sphere edge.
  idx :
      A scalar element index

  Returns
  -------
  tuple[int, int, int]
      (face_idx, horizontal_idx, vertical_idx)
      of the element.

  Notes
  -----
  This should be the inverse of elem_id_fn
  """
  face_id = int(idx / nx**2)
  x_id = int((idx - face_id * nx**2) / nx)
  y_id = int(idx - face_id * nx**2 - x_id * nx)
  return face_id, x_id, y_id


def gen_cube_topo(nx):
  """
  Generates the cartesian coordinates and topological
  connectivity of a quasi-regular grid
  on the cubed sphere.

  Parameters
  ----------
  nx : int
      the 1st param name `first`

  Returns
  -------
  face_connectivity: Array[tuple[elem_idx, edge_idx, 3], Int]
    An array containing the topological information about the grid.
    It is unpacked as
    (remote_elem_idx, remote_edge_idx, same_direction) = face_connectivity[local_elem_idx,
                                                                           edge_idx, :]
  face_mask: Array[tuple[elem_idx], Int]
    An integer mask describing which face of the cubed sphere each element lies on.
  face_position: Array[tuple[elem_idx, vert_idx, xyz], Float]
    Positions of the element vertices on the reference cube 
    in 3d Cartesian space
  face_position_2d: Array[tuple[elem_idx, vert_idx, xy], Float]
    Positions of the element vertices within the local (x, y)
    coordinates on the cubed-sphere face that contains it.
  """

  NFACE = 6 * nx**2
  face_connectivity = np.zeros(shape=(NFACE, 4, 3), dtype=np.int64)
  face_position = np.zeros(shape=(NFACE, 4, 3), dtype=np.float64)
  face_position_2d = np.zeros(shape=(NFACE, 4, 2), dtype=np.float64)
  face_mask = np.zeros(NFACE, np.int32)

  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
    for x_idx in range(nx):
      for y_idx in range(nx):
        face_mask[elem_id_fn(nx, face_idx, x_idx, y_idx)] = face_idx
        corner_list = vert_info[face_idx]
        x_frac_left = x_idx / nx
        x_frac_right = (x_idx + 1) / nx
        y_frac_top = y_idx / nx
        y_frac_bottom = (y_idx + 1) / nx
        vec_top = corner_list[1] - corner_list[0]
        vec_left = corner_list[2] - corner_list[0]
        element_corners = [corner_list[0] + x_frac_left * vec_top + y_frac_top * vec_left,
                           corner_list[0] + x_frac_right * vec_top + y_frac_top * vec_left,
                           corner_list[0] + x_frac_left * vec_top + y_frac_bottom * vec_left,
                           corner_list[0] + x_frac_right * vec_top + y_frac_bottom * vec_left]
        # face_x_idx, face_y_idx, edge_idx, edge_direction
        top_info = [elem_id_fn(nx, face_idx, x_idx, y_idx - 1), BOTTOM_EDGE, FORWARDS]
        left_info = [elem_id_fn(nx, face_idx, x_idx - 1, y_idx), RIGHT_EDGE, FORWARDS]
        right_info = [elem_id_fn(nx, face_idx, x_idx + 1, y_idx), LEFT_EDGE, FORWARDS]
        bottom_info = [elem_id_fn(nx, face_idx, x_idx, y_idx + 1), TOP_EDGE, FORWARDS]
        if x_idx == 0:
          edge_idx = LEFT_EDGE
          free_idx = y_idx
          face_pair, edge_pair, edge_dir = face_topo[face_idx][edge_idx]
          x_idx_out, y_idx_out = edge_match(nx, free_idx, edge_pair, edge_dir)
          left_info = [elem_id_fn(nx, face_pair, x_idx_out, y_idx_out), edge_pair, edge_dir]
        if x_idx == nx - 1:
          edge_idx = RIGHT_EDGE
          free_idx = y_idx
          face_pair, edge_pair, edge_dir = face_topo[face_idx][edge_idx]
          x_idx_out, y_idx_out = edge_match(nx, free_idx, edge_pair, edge_dir)
          right_info = [elem_id_fn(nx, face_pair, x_idx_out, y_idx_out), edge_pair, edge_dir]
        if y_idx == nx - 1:
          edge_idx = BOTTOM_EDGE
          free_idx = x_idx
          face_pair, edge_pair, edge_dir = face_topo[face_idx][edge_idx]
          x_idx_out, y_idx_out = edge_match(nx, free_idx, edge_pair, edge_dir)
          bottom_info = [elem_id_fn(nx, face_pair, x_idx_out, y_idx_out), edge_pair, edge_dir]
        if y_idx == 0:
          edge_idx = TOP_EDGE
          free_idx = x_idx
          face_pair, edge_pair, edge_dir = face_topo[face_idx][edge_idx]
          x_idx_out, y_idx_out = edge_match(nx, free_idx, edge_pair, edge_dir)
          top_info = [elem_id_fn(nx, face_pair, x_idx_out, y_idx_out), edge_pair, edge_dir]

        elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
        face_connectivity[elem_idx, TOP_EDGE, :] = top_info
        face_connectivity[elem_idx, BOTTOM_EDGE, :] = bottom_info
        face_connectivity[elem_idx, LEFT_EDGE, :] = left_info
        face_connectivity[elem_idx, RIGHT_EDGE, :] = right_info
        for v_idx, corner in enumerate(element_corners):
          face_position[elem_idx, v_idx, :] = corner
          face_position_2d[elem_idx, v_idx, 0] = corner[axis_info[face_idx][0]] * axis_info[face_idx][1]
          face_position_2d[elem_idx, v_idx, 1] = corner[axis_info[face_idx][2]] * axis_info[face_idx][3]

  return face_connectivity, face_mask, face_position, face_position_2d


def gen_vert_redundancy(nx, face_connectivity, face_position):
  """
  Enumerate redundant DOFs on the elemental
  representation of a quasi-regular cubed sphere grid.

  Parameters
  ----------
  nx : int
    The number of elements on a cubed-sphere edge.
  face_connectivity : Array[tuple[elem_idx, edge_idx, 3], Int]
    An array containing the topological information about the grid.
    It is unpacked as
    (remote_elem_idx, remote_edge_idx, same_direction) = face_connectivity[local_elem_idx,
                                                                           edge_idx, :]

  Returns
  -------
  vert_redundancy: dict[local_elem_idx, dict[vert_idx, set[tuple[remote_elem_idx, vert_idx]]]]
      dict[local_elem_idx][vert_idx] is a set of tuples
      (remote_elem_idx, vert_idx_pair) which represent vertices that
      share the same physical coordinates as (local_elem_idx, vert_idx).
      Therefore, they represent redundant degrees of freedom.

  Notes
  -----
  This struct deliberately does not contain diagonal associations, i.e.
  (local_elem_idx, vert_idx) <-/-> (local_elem_idx, vert_idx)
  """
  vert_redundancy = dict()

  def wrap(elem_idx, vert_idx):
    if elem_idx not in vert_redundancy.keys():
      vert_redundancy[elem_idx] = dict()
    if vert_idx not in vert_redundancy[elem_idx].keys():
      vert_redundancy[elem_idx][vert_idx] = set()

  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
    for x_idx in range(nx):
      for y_idx in range(nx):
        for edge_idx in [TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE]:
          elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
          idx_pair, edge_idx_pair, is_forwards = face_connectivity[elem_idx, edge_idx, :]
          face_idx_pair, x_idx_pair, y_idx_pair = inv_elem_id_fn(nx, idx_pair)
          v0_local, v1_local = edge_to_vert(edge_idx)
          v0_pair, v1_pair = edge_to_vert(edge_idx_pair, is_forwards=is_forwards)
          wrap(elem_idx, v0_local)
          wrap(elem_idx, v1_local)
          vert_redundancy[elem_idx][v0_local].add((idx_pair, v0_pair))
          vert_redundancy[elem_idx][v1_local].add((idx_pair, v1_pair))
  # The following is a crude-but-concise way to ensure
  # (elem_idx_pair, v_idx_pair) in vert_redundancy[elem_idx][v_idx]  <=>
  # (elem_idx, v_idx) in vert_redundancy[elem_idx_pair][v_idx_pair]
  for _ in range(MAX_VERT_DEGREE):
    for elem_idx in vert_redundancy.keys():
      for vert_idx in vert_redundancy[elem_idx].keys():
        for elem_idx_pair, vert_idx_pair in vert_redundancy[elem_idx][vert_idx]:
          vert_redundancy[elem_idx_pair][vert_idx_pair].update(vert_redundancy[elem_idx][vert_idx])
  # filter out diagonal
  for elem_idx in vert_redundancy.keys():
    for vert_idx in vert_redundancy[elem_idx].keys():
       if (elem_idx, vert_idx) in vert_redundancy[elem_idx][vert_idx]:
        vert_redundancy[elem_idx][vert_idx].remove((elem_idx, vert_idx))

  return vert_redundancy
