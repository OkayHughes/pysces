from ..config import jnp, jit, flip, np
from functools import partial


@jit
def sphere_gradient(f, grid, a=1.0):
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
  df_da = jnp.einsum("fij,ki->fkj", f, grid["deriv"])
  df_db = jnp.einsum("fij,kj->fik", f, grid["deriv"])
  df_dab = jnp.stack((df_da, df_db), axis=-1)
  return 1.0 / a * flip(jnp.einsum("fijg,fijgs->fijs", df_dab, grid["jacobian_inv"]), -1)


@jit
def sphere_divergence(u, grid, a=1.0):
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
  u_contra = 1.0 / a * grid["met_det"][:, :, :, np.newaxis] * sph_to_contra(u, grid)
  du_da = jnp.einsum("fij,ki->fkj", u_contra[:, :, :, 0], grid["deriv"])
  du_db = jnp.einsum("fij,kj->fik", u_contra[:, :, :, 1], grid["deriv"])
  div = grid["recip_met_det"][:, :, :] * (du_da + du_db)
  return div


@jit
def sphere_vorticity(u, grid, a=1.0):
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
  u_cov = sph_to_cov(u, grid)
  dv_da = jnp.einsum("fij,ki->fkj", u_cov[:, :, :, 1], grid["deriv"])
  du_db = jnp.einsum("fij,kj->fik", u_cov[:, :, :, 0], grid["deriv"])
  vort = 1.0 / a * grid["recip_met_det"][:, :, :] * (du_db - dv_da)
  return vort


@jit
def sphere_laplacian(f, grid, a=1.0):
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
  grad = sphere_gradient(f, grid, a=a)
  return sphere_divergence(grad, grid, a=a)


@jit
def sphere_laplacian_wk(f, grid, a=1.0):
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
  grad = sphere_gradient(f, grid, a=a)
  return sphere_divergence_wk(grad, grid, a=a)


@jit
def sphere_gradient_wk_cov(s, grid, a=1.0):
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
  gll_weights = grid["gll_weights"]
  deriv = grid["deriv"]
  met_inv = grid["met_inv"]
  met_det = grid["met_det"]
  ds_contra_term_1 = - jnp.einsum("j,n,fmn,fmn,fjn,jm->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 0, 0],
                                  met_det, s, deriv)
  ds_contra_term_2 = - jnp.einsum("m,j,fmn,fmn,fmj,jn->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 1, 0],
                                  met_det, s, deriv)
  ds_contra_term_3 = - jnp.einsum("j,n,fmn,fmn,fjn,jm->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 0, 1],
                                  met_det, s, deriv)
  ds_contra_term_4 = - jnp.einsum("m,j,fmn,fmn,fmj,jn->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 1, 1],
                                  met_det, s, deriv)
  ds_contra = jnp.stack((ds_contra_term_1 + ds_contra_term_2,
                         ds_contra_term_3 + ds_contra_term_4), axis=-1)
  return 1.0 / a * contra_to_sph(ds_contra, grid)


@jit
def sphere_curl_wk_cov(s, grid, a=1.0):
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
  gll_weights = grid["gll_weights"]
  deriv = grid["deriv"]
  ds_contra = jnp.stack((jnp.einsum("m,j,fmj,jn->fmn", gll_weights, gll_weights, s, deriv),
                         -jnp.einsum("j,n,fjn,jm->fmn", gll_weights, gll_weights, s, deriv)), axis=-1)
  return 1.0 / a * contra_to_sph(ds_contra, grid)


@partial(jit, static_argnames=["damp"])
def sphere_vec_laplacian_wk(u, grid, a=1.0, nu_div_fact=1.0, damp=False):
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
  div = sphere_divergence(u, grid, a=a) * nu_div_fact
  vor = sphere_vorticity(u, grid, a=a)
  laplacian = sphere_gradient_wk_cov(div, grid, a=a) - sphere_curl_wk_cov(vor, grid, a=a)
  gll_weights = grid["gll_weights"]
  if damp:
    out = laplacian + jnp.stack((2 * (gll_weights[np.newaxis, :, np.newaxis] *
                                      gll_weights[np.newaxis, np.newaxis, :] *
                                      grid["met_det"] * u[:, :, :, 0] * (1 / a)**2),
                                     (gll_weights[np.newaxis, :, np.newaxis] *
                                      gll_weights[np.newaxis, np.newaxis, :] *
                                      grid["met_det"] * u[:, :, :, 1] * (1 / a)**2)), axis=-1)
  else:
    out = laplacian
  return out


@jit
def sphere_divergence_wk(u, grid, a=1.0):
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
  contra = sph_to_contra(u, grid)
  gll_weights = grid["gll_weights"]
  met_det = grid["met_det"]
  deriv = grid["deriv"]
  du_da_wk = - jnp.einsum("n,j,fjn,fjn,jm->fmn", gll_weights, gll_weights, met_det, contra[:, :, :, 0], deriv)
  du_db_wk = - jnp.einsum("m,j,fmj,fmj,jn->fmn", gll_weights, gll_weights, met_det, contra[:, :, :, 1], deriv)
  return 1.0 / a * (du_da_wk + du_db_wk)


@jit
def contra_to_sph(u, grid):
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
  return flip(jnp.einsum("fijg,fijsg->fijs", u, grid["jacobian"]), -1)


@jit
def sph_to_contra(u, grid):
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
  return jnp.einsum("fijs,fijgs->fijg", flip(u, -1), grid["jacobian_inv"])


@jit
def sph_to_cov(u, grid):
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
  return jnp.einsum("fijs,fijsg->fijg", flip(u, -1
                                             ), grid["jacobian"])


@jit
def inner_prod(f, g, grid):
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
  return jnp.sum(f * g * (grid["met_det"] *
                          grid["gll_weights"][np.newaxis, :, np.newaxis] *
                          grid["gll_weights"][np.newaxis, np.newaxis, :]))
