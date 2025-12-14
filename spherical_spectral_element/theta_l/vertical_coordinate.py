from ..config import jit, np


def create_vertical_grid(hybrid_a_i, hybrid_b_i, p0):
  v_grid = {"reference_pressure": p0,
            "hybrid_a_i": hybrid_a_i,
            "hybrid_b_i": hybrid_b_i}
  v_grid["hybrid_a_m"] = 0.5 * (hybrid_a_i[1:] + hybrid_a_i[:-1])
  v_grid["hybrid_b_m"] = 0.5 * (hybrid_b_i[1:] + hybrid_b_i[:-1])
  return v_grid


@jit
def mass_from_coordinate_midlev(ps, v_grid):
  return (v_grid["reference_pressure"] * v_grid["hybrid_a_m"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])


@jit
def dmass_from_coordinate(ps, v_grid):
  da = (v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, 1:] -
        v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, :-1])
  db = (v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, 1:] -
        v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, :-1])
  return (v_grid["reference_pressure"] * da +
          db * ps[:, :, :, np.newaxis])


@jit
def mass_from_coordinate_interface(ps, v_grid):
  return (v_grid["reference_pressure"] * v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])
