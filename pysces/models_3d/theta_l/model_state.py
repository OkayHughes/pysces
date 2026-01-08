from ...config import jnp, jit, np, flip, vmap_1d_apply
from ...operations_2d.assembly import project_scalar, project_scalar_for
from ...distributed_memory.multiprocessing import project_scalar_triple_mpi
from ...operations_2d.operators import sphere_gradient
from ..vertical_remap import zerroukat_remap
from ..mass_coordinate import dmass_from_coordinate, mass_from_coordinate_midlev
from .eqn_of_state import get_balanced_phi, get_p_mid
from ..utils_3d import get_delta, get_surface_sum, g_from_phi
from functools import partial


@jit
def wrap_model_struct(u, vtheta_dpi, dpi, phi_surf, grad_phi_surf, phi_i, w_i):
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
  state = {"u": u,
           "vtheta_dpi": vtheta_dpi,
           "dpi": dpi,
           "phi_surf": phi_surf,
           "grad_phi_surf": grad_phi_surf,
           "phi_i": phi_i,
           "w_i": w_i
           }
  return state


@jit
def wrap_tracer_avg_struct(avg_u, avg_dpi, avg_dpi_dissip):
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
  return {"avg_v": avg_u,
          "avg_dpi": avg_dpi,
          "avg_dpi_dissip": avg_dpi_dissip}


@partial(jit, static_argnames=["dims"])
def init_model_struct(u, vtheta_dpi, dpi, phi_surf, phi_i, w_i, h_grid, dims, config):
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
  grad_phi_surf_discont = sphere_gradient(phi_surf, h_grid, a=config["radius_earth"])
  grad_phi_surf = jnp.stack((project_scalar(grad_phi_surf_discont[:, :, :, 0], h_grid, dims),
                             project_scalar(grad_phi_surf_discont[:, :, :, 1], h_grid, dims)), axis=-1)
  state = {"u": u,
           "vtheta_dpi": vtheta_dpi,
           "dpi": dpi,
           "phi_surf": phi_surf,
           "grad_phi_surf": grad_phi_surf,
           "phi_i": phi_i,
           "w_i": w_i
           }
  return state


def init_tracer_struct(Q):
  return {"Q": Q}

# TODO 12/23/25: add wrapper functions that apply
# summation, and project_scalar so model interface remains identical.
# also: refactor into separate file, since these are non-jittable


@partial(jit, static_argnames=["dims", "scaled"])
def project_scalar_3d(variable, h_grid, dims, scaled=True):
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
  def project_onlyarg(vec):
    return project_scalar(vec, h_grid, dims, scaled=scaled)
  return vmap_1d_apply(project_onlyarg, variable, -1, -1)

# @partial(jit, static_argnames=["dims", "scaled"])
# def project_scalar_3d(variable, h_grid, dims, scaled=True):
#   return project_scalar_triple_mpi([variable], h_grid, dims, scaled=scaled, two_d=False)[0]


def project_scalar_3d_for(variable, h_grid, dims, scaled=True):
  levs = []
  for lev_idx in range(variable.shape[-1]):
    levs.append(project_scalar_for(variable[:, :, :, lev_idx], h_grid))
  return np.stack(levs, axis=-1)


@partial(jit, static_argnames=["dims", "scaled", "hydrostatic"])
def project_model_state(state_in, h_grid, dims, scaled=True, hydrostatic=True):
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
  u_cont = project_scalar_3d(state_in["u"][:, :, :, :, 0], h_grid, dims, scaled=scaled)
  v_cont = project_scalar_3d(state_in["u"][:, :, :, :, 1], h_grid, dims, scaled=scaled)
  vtheta_dpi_cont = project_scalar_3d(state_in["vtheta_dpi"][:, :, :, :], h_grid, dims, scaled=scaled)
  dpi_cont = project_scalar_3d(state_in["dpi"][:, :, :, :], h_grid, dims, scaled=scaled)
  if hydrostatic:
    w_i_cont = state_in["w_i"]
    phi_i_cont = state_in["phi_i"]
  else:
    w_i_cont = project_scalar_3d(state_in["w_i"], h_grid, dims, scaled=scaled)
    phi_i_cont = project_scalar_3d(state_in["phi_i"], h_grid, dims, scaled=scaled)
  return wrap_model_struct(jnp.stack((u_cont, v_cont), axis=-1),
                           vtheta_dpi_cont, dpi_cont, state_in["phi_surf"],
                           state_in["grad_phi_surf"],
                           phi_i_cont,
                           w_i_cont)


@jit
def pi_surf_from_state(state_in, v_grid):
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
  return jnp.sum(state_in["dpi"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"]


@partial(jit, static_argnames=["hydrostatic", "deep", "num_lev"])
def remap_state(state_in, v_grid, config, num_lev, hydrostatic=True, deep=False):
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
  pi_surf = pi_surf_from_state(state_in, v_grid)
  dpi_ref = dmass_from_coordinate(pi_surf,
                                  v_grid)
  p_mid = get_p_mid(state_in, v_grid, config)
  dpi = state_in["dpi"]
  u_model = state_in["u"][:, :, :, :, 0] * dpi
  v_model = state_in["u"][:, :, :, :, 1] * dpi
  vtheta_dpi = state_in["vtheta_dpi"]
  if not hydrostatic:
    phi_ref = get_balanced_phi(state_in["phi_surf"],
                               p_mid,
                               state_in["vtheta_dpi"],
                               config)
    phi_pert = state_in["phi_i"] - phi_ref
    dphi = get_delta(phi_pert)
    dw = get_delta(state_in["w_i"])
    Qdp = jnp.stack([u_model, v_model, vtheta_dpi,
                     dphi, dw], axis=-1)
  else:
    Qdp = jnp.stack([u_model, v_model, vtheta_dpi], axis=-1)
  Qdp_out = zerroukat_remap(Qdp, state_in["dpi"], dpi_ref, num_lev, filter=True)
  u_remap = jnp.stack((Qdp_out[:, :, :, :, 0] / dpi_ref,
                       Qdp_out[:, :, :, :, 1] / dpi_ref), axis=-1)
  vtheta_dpi_remap = Qdp_out[:, :, :, :, 2]
  if not hydrostatic:
    p_mid = mass_from_coordinate_midlev(pi_surf, v_grid)
    phi_ref_new = get_balanced_phi(state_in["phi_surf"],
                                   p_mid,
                                   vtheta_dpi_remap,
                                   config)
    phi_i_remap = get_surface_sum(-Qdp_out[:, :, :, :, 3], jnp.zeros_like(state_in["phi_surf"])) + phi_ref_new
    w_i_surf = ((u_remap[:, :, :, -1, 0] * state_in["grad_phi_surf"][:, :, :, 0] +
                 u_remap[:, :, :, -1, 1] * state_in["grad_phi_surf"][:, :, :, 1]) /
                g_from_phi(state_in["phi_surf"], config, deep=deep))
    w_i_upper = flip(jnp.cumsum(-flip(Qdp[:, :, :, :, 4], -1), axis=-1), -1) + state_in["w_i"][:, :, :, -1:]
    w_i_remap = jnp.concatenate((w_i_upper, w_i_surf[:, :, :, np.newaxis]), axis=-1)
  else:
    phi_i_remap = state_in["phi_i"]
    w_i_remap = state_in["w_i"]
  return wrap_model_struct(u_remap,
                           vtheta_dpi_remap,
                           dpi_ref,
                           state_in["phi_surf"],
                           state_in["grad_phi_surf"],
                           phi_i_remap,
                           w_i_remap)
