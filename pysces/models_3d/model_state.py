from ..config import jnp, jit, flip, np
from functools import partial
from ..operations_2d.operators import horizontal_gradient
from ..distributed_memory.global_assembly import project_scalar_global
from .model_info import f_plane_models, deep_atmosphere_models, thermodynamic_variable_names, hydrostatic_models, cam_se_models, homme_models, variable_kappa_models
from .mass_coordinate import d_mass_from_coordinate, mass_from_coordinate_midlev
from .homme.thermodynamics import get_balanced_phi, get_p_mid
from .utils_3d import get_delta, get_surface_sum, g_from_phi
from .vertical_remap import zerroukat_remap


@jit
def sum_simple_tracers(state1, state2, fold_coeff1, fold_coeff2):
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
  state_out = {}
  for tracer_name in state1.keys():
    state_out[tracer_name] = fold_coeff1 * state1[tracer_name] + fold_coeff2 * state2[tracer_name]
  return state_out


def advance_simple_tracers(tracer_states, coeffs, model):
  if model in variable_kappa_models:
    raise NotImplementedError("I'm still working on dry air tracers")
  moisture_species = sum_simple_tracers(tracer_states[0]["moisture_species"],
                                        tracer_states[1]["moisture_species"],
                                        coeffs[0],
                                        coeffs[1])
  passiveish_tracers = sum_simple_tracers(tracer_states[0]["tracers"],
                                          tracer_states[1]["tracers"],
                                          coeffs[0],
                                          coeffs[1])
  for coeff_idx in range(2, len(tracer_states)):
    moisture_species = sum_simple_tracers(moisture_species,
                                          tracer_states[coeff_idx]["moisture_species"],
                                          coeffs[0],
                                          coeffs[1])
    passiveish_tracers = sum_simple_tracers(passiveish_tracers,
                                            tracer_states[coeff_idx]["tracers"],
                                            1.0,
                                            coeffs[coeff_idx])

  return wrap_tracer_struct(moisture_species, passiveish_tracers)


@jit
def wrap_tracer_avg_struct(avg_u, avg_d_mass, avg_d_mass_dissip):
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
          "avg_d_mass": avg_d_mass,
          "avg_d_mass_dissip": avg_d_mass_dissip}


@partial(jit, static_argnames="model")
def wrap_dynamics_struct(u, thermodynamic_variable, d_mass, model, phi_i=None, w_i=None):
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
           thermodynamic_variable_names[model]: thermodynamic_variable,
           "d_mass": d_mass
           }
  if phi_i is not None:
    state["phi_i"] = phi_i
  if w_i is not None:
    state["w_i"] = w_i
  return state


def wrap_model_state(dynamics, static_forcing, tracers):
  return {"dynamics": dynamics,
          "static_forcing": static_forcing,
          "tracers": tracers}


def wrap_tracer_struct(moisture_species, tracers, dry_species=None):
  tracer_struct = {"moisture_species": moisture_species,
                   "tracers": tracers}
  if dry_species is not None:
    tracer_struct["dry_species"] = dry_species
  return tracer_struct


def wrap_static_forcing(phi_surf, grad_phi_surf, coriolis_param, nontrad_coriolis_param=None):
  static_forcing = {"phi_surf": phi_surf,
                    "grad_phi_surf": grad_phi_surf,
                    "coriolis_param": coriolis_param}
  if nontrad_coriolis_param is not None:
    static_forcing["nontrad_coriolis_param"] = nontrad_coriolis_param
  return static_forcing


def init_static_forcing(phi_surf, h_grid, physics_config, dims, model, f_plane_center=jnp.pi/4.0):
  grad_phi_surf_discont = horizontal_gradient(phi_surf, h_grid, a=physics_config["radius_earth"])
  grad_phi_surf = jnp.stack([project_scalar_global([grad_phi_surf_discont[:, :, :, 0]], h_grid, dims)[0],
                             project_scalar_global([grad_phi_surf_discont[:, :, :, 1]], h_grid, dims)[0]], axis=-1)
  if model in f_plane_models:
    coriolis_param = 2.0 * physics_config["period_earth"] * jnp.sin(f_plane_center) * jnp.ones_like(h_grid["physical_coords"][:, :, :, 0])
  else:
    coriolis_param = 2.0 * physics_config["period_earth"] * jnp.sin(h_grid["physical_coords"][:, :, :, 0])
  if model in deep_atmosphere_models:
    nontrad_coriolis_param = 2.0 * physics_config["period_earth"] * jnp.cos(h_grid["physical_coords"][:, :, :, 0])
  else:
    nontrad_coriolis_param = None
  return wrap_static_forcing(phi_surf, grad_phi_surf, coriolis_param, nontrad_coriolis_param=nontrad_coriolis_param)


@partial(jit, static_argnames=["dims", "scaled", "model"])
def project_dynamics_state(dynamics_in, h_grid, dims, model, scaled=True):
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
  u_cont = project_scalar_3d(dynamics_in["u"][:, :, :, :, 0], h_grid, dims, scaled=scaled)
  v_cont = project_scalar_3d(dynamics_in["u"][:, :, :, :, 1], h_grid, dims, scaled=scaled)
  thermo_var_cont = project_scalar_3d(dynamics_in[thermodynamic_variable_names[model]][:, :, :, :], h_grid, dims, scaled=scaled)
  d_mass_cont = project_scalar_3d(dynamics_in["d_mass"][:, :, :, :], h_grid, dims, scaled=scaled)
  if model not in hydrostatic_models:
    w_i_cont = project_scalar_3d(dynamics_in["w_i"], h_grid, dims, scaled=scaled)
    phi_i_cont = project_scalar_3d(dynamics_in["phi_i"], h_grid, dims, scaled=scaled)
  else:
    phi_i_cont = None
    w_i_cont = None
  return wrap_dynamics_struct(jnp.stack((u_cont, v_cont), axis=-1),
                              thermo_var_cont,
                              d_mass_cont,
                              model,
                              phi_i=phi_i_cont,
                              w_i=w_i_cont)


@partial(jit, static_argnames=["dims", "scaled"])
def project_scalar_3d(variable, h_grid, dims, scaled=True):
  return project_scalar_global([variable], h_grid, dims, scaled=scaled, two_d=False)[0]


@jit
def surface_mass_from_state(state_in, v_grid):
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
  return jnp.sum(state_in["d_mass"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]



@partial(jit, static_argnames=["num_lev", "model"])
def remap_dynamics(dynamics_in, static_forcing, v_grid, physics_config, num_lev, model):
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
  pi_surf = surface_mass_from_state(dynamics_in, v_grid)
  d_mass_ref = d_mass_from_coordinate(pi_surf,
                                     v_grid)
  d_mass = dynamics_in["d_mass"]
  u_model = dynamics_in["u"][:, :, :, :, 0] * d_mass
  v_model = dynamics_in["u"][:, :, :, :, 1] * d_mass
  if model in cam_se_models:
    thermo_model = dynamics_in["T"] * d_mass
  else:
    thermo_model = dynamics_in["theta_v_d_mass"]
  if model not in hydrostatic_models:
    p_mid = get_p_mid(dynamics_in, v_grid, physics_config)
    phi_ref = get_balanced_phi(static_forcing["phi_surf"],
                               p_mid,
                               dynamics_in["theta_v_d_mass"],
                               physics_config)
    phi_pert = dynamics_in["phi_i"] - phi_ref
    d_phi = get_delta(phi_pert)
    dw = get_delta(dynamics_in["w_i"])
    Qdp = jnp.stack([u_model, v_model, thermo_model,
                     d_phi, dw], axis=-1)
  else:
    Qdp = jnp.stack([u_model, v_model, thermo_model], axis=-1)
  Qdp_out = zerroukat_remap(Qdp, dynamics_in["d_mass"], d_mass_ref, num_lev, filter=True)
  u_remap = jnp.stack((Qdp_out[:, :, :, :, 0] / d_mass_ref,
                       Qdp_out[:, :, :, :, 1] / d_mass_ref), axis=-1)
  if model in cam_se_models:
    thermo_remap = Qdp_out[:, :, :, :, 2] / d_mass_ref
  else:
    thermo_remap = Qdp_out[:, :, :, :, 2]
  
  if model not in hydrostatic_models:
    p_mid = mass_from_coordinate_midlev(pi_surf, v_grid)
    phi_ref_new = get_balanced_phi(static_forcing["phi_surf"],
                                   p_mid,
                                   thermo_remap,
                                   physics_config)
    phi_i_remap = get_surface_sum(-Qdp_out[:, :, :, :, 3], jnp.zeros_like(static_forcing["phi_surf"])) + phi_ref_new
    w_i_surf = ((u_remap[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
                 u_remap[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
                g_from_phi(static_forcing["phi_surf"], physics_config, model))
    w_i_upper = flip(jnp.cumsum(-flip(Qdp[:, :, :, :, 4], -1), axis=-1), -1) + dynamics_in["w_i"][:, :, :, -1:]
    w_i_remap = jnp.concatenate((w_i_upper, w_i_surf[:, :, :, np.newaxis]), axis=-1)
  else:
    phi_i_remap = None
    w_i_remap = None
  return wrap_dynamics_struct(u_remap,
                              thermo_remap,
                              d_mass_ref,
                              model,
                              phi_i=phi_i_remap,
                              w_i=w_i_remap)


@jit
def sum_dynamics_states(state1, state2, fold_coeff1, fold_coeff2, model):
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
  if model not in hydrostatic_models:
    phi_i = state1["phi_i"] * fold_coeff1 + state2["phi_i"] * fold_coeff2
    w_i = state1["w_i"] * fold_coeff1 + state2["w_i"] * fold_coeff2
  else:
    phi_i = None
    w_i = None
  thermo_var_name = thermodynamic_variable_names[model]
  return wrap_dynamics_struct(state1["u"] * fold_coeff1 + state2["u"] * fold_coeff2,
                              state1[thermo_var_name] * fold_coeff1 + state2[thermo_var_name] * fold_coeff2,
                              state1["d_mass"] * fold_coeff1 + state2["d_mass"] * fold_coeff2,
                              model,
                              phi_i=phi_i,
                              w_i=w_i)


@jit
def advance_dynamics(states, coeffs, model):
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
  state_out = sum_dynamics_states(states[0],
                             states[1],
                             coeffs[0],
                             coeffs[1], model)
  for coeff_idx in range(2, len(states)):
    state_out = sum_dynamics_states(state_out,
                               states[coeff_idx],
                               1.0,
                               coeffs[coeff_idx], model)
  return state_out


