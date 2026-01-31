from ..config import partial, jit, vmap_1d_apply
from ..operations_2d.operators import horizontal_divergence, horizontal_vorticity
from ..operations_2d.operators import horizontal_gradient, horizontal_weak_laplacian, horizontal_weak_vector_laplacian


@jit
def horizontal_divergence_3d(vector,
                             h_grid,
                             physics_config):
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
  sph_op = partial(horizontal_divergence, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -1)


@jit
def horizontal_vorticity_3d(vector,
                            h_grid,
                            physics_config):
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
  sph_op = partial(horizontal_vorticity, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -1)


@jit
def horizontal_weak_laplacian_3d(scalar,
                                 h_grid,
                                 physics_config):
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
  sph_op = partial(horizontal_weak_laplacian, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@jit
def horizontal_weak_vector_laplacian_3d(vector,
                                        h_grid,
                                        physics_config):
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
  sph_op = partial(horizontal_weak_vector_laplacian, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -2)


@jit
def horizontal_gradient_3d(scalar,
                           h_grid,
                           physics_config):
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
  sph_op = partial(horizontal_gradient, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, scalar, -1, -2)
