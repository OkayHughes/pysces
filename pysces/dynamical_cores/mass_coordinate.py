from ..config import jit, np
from ..model_info import moist_mixing_ratio_models


def create_vertical_grid(hybrid_a_i, hybrid_b_i, reference_surface_mass, model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  v_grid = {"reference_surface_mass": reference_surface_mass,
            "hybrid_a_i": hybrid_a_i,
            "hybrid_b_i": hybrid_b_i}
  v_grid["hybrid_a_m"] = 0.5 * (hybrid_a_i[1:] + hybrid_a_i[:-1])
  v_grid["hybrid_b_m"] = 0.5 * (hybrid_b_i[1:] + hybrid_b_i[:-1])
  if model in moist_mixing_ratio_models:
    v_grid["moist"] = 1.0
  else:
    v_grid["dry"] = 1.0
  return v_grid


@jit
def mass_from_coordinate_midlev(ps, v_grid):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  return (v_grid["reference_surface_mass"] * v_grid["hybrid_a_m"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])


@jit
def d_mass_from_coordinate(ps, v_grid):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  da = (v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, 1:] -
        v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, :-1])
  db = (v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, 1:] -
        v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, :-1])
  return (v_grid["reference_surface_mass"] * da +
          db * ps[:, :, :, np.newaxis])


@jit
def mass_from_coordinate_interface(ps, v_grid):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  return (v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])
