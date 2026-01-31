from ..config import jnp, jit, flip, np
from functools import partial
from ..operations_2d.operators import horizontal_gradient
from ..distributed_memory.global_assembly import project_scalar_global
from ..model_info import (f_plane_models,
                          deep_atmosphere_models,
                          thermodynamic_variable_names,
                          hydrostatic_models,
                          cam_se_models,
                          moist_mixing_ratio_models)
from .mass_coordinate import surface_mass_to_d_mass, surface_mass_to_midlevel_mass
from .homme.thermodynamics import eval_balanced_geopotential, eval_midlevel_pressure
from .utils_3d import interface_to_delta, cumulative_sum, phi_to_g
from .vertical_remap import zerroukat_remap
from ..distributed_memory.global_communication import global_sum


@partial(jit, static_argnames=["is_dry_air_species"])
def sum_tracers(state1,
                state2,
                fold_coeff1,
                fold_coeff2,
                is_dry_air_species=False):
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


@partial(jit, static_argnames=["model"])
def advance_tracers(tracer_states,
                    coeffs,
                    model):
  moisture_species = sum_tracers(tracer_states[0]["moisture_species"],
                                 tracer_states[1]["moisture_species"],
                                 coeffs[0],
                                 coeffs[1])
  passiveish_tracers = sum_tracers(tracer_states[0]["tracers"],
                                   tracer_states[1]["tracers"],
                                   coeffs[0],
                                   coeffs[1])
  if model in cam_se_models:
    dry_air_species = sum_tracers(tracer_states[0]["dry_air_species"],
                                  tracer_states[1]["dry_air_species"],
                                  coeffs[0],
                                  coeffs[1],
                                  is_dry_air_species=True)
  else:
    dry_air_species = None

  for coeff_idx in range(2, len(tracer_states)):
    moisture_species = sum_tracers(moisture_species,
                                   tracer_states[coeff_idx]["moisture_species"],
                                   coeffs[0],
                                   coeffs[1])
    passiveish_tracers = sum_tracers(passiveish_tracers,
                                     tracer_states[coeff_idx]["tracers"],
                                     1.0,
                                     coeffs[coeff_idx])
    if model in cam_se_models:
      dry_air_species = sum_tracers(dry_air_species,
                                    tracer_states[coeff_idx]["dry_air_species"],
                                    1.0,
                                    coeffs[coeff_idx],
                                    is_dry_air_species=True)

  return wrap_tracers(moisture_species,
                      passiveish_tracers,
                      model,
                      dry_air_species=dry_air_species)


@jit
def wrap_tracer_avg(avg_u,
                    avg_d_mass,
                    avg_d_mass_dissip):
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


@partial(jit, static_argnames=["model"])
def wrap_dynamics(u,
                  thermodynamic_variable,
                  d_mass,
                  model,
                  phi_i=None,
                  w_i=None):
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


@jit
def wrap_model_state(dynamics,
                     static_forcing,
                     tracers):
  return {"dynamics": dynamics,
          "static_forcing": static_forcing,
          "tracers": tracers}


@partial(jit, static_argnames=["model"])
def copy_dynamics(dynamics,
                  model):
  if model not in hydrostatic_models:
    phi_i = dynamics["phi_i"]
    w_i = dynamics["w_i"]
  else:
    phi_i = None
    w_i = None
  return wrap_dynamics(jnp.copy(dynamics["u"]),
                       jnp.copy(dynamics[thermodynamic_variable_names[model]]),
                       jnp.copy(dynamics["d_mass"]),
                       model,
                       phi_i=phi_i,
                       w_i=w_i)


@partial(jit, static_argnames=["model"])
def copy_tracers(tracers,
                 model):
  if model in cam_se_models:
    dry_air_species = {}
    for species_name in tracers["dry_air_species"].keys():
      dry_air_species[species_name] = jnp.copy(tracers["dry_air_species"][species_name])
  else:
    dry_air_species = None
  moisture_species = {}
  for species_name in tracers["moisture_species"].keys():
    moisture_species[species_name] = jnp.copy(tracers["moisture_species"][species_name])
  tracers_new = {}
  for species_name in tracers["tracers"].keys():
    tracers_new[species_name] = jnp.copy(tracers["tracers"][species_name])
  return wrap_tracers(moisture_species,
                      tracers_new,
                      model,
                      dry_air_species=dry_air_species)


@partial(jit, static_argnames=["model"])
def copy_model_state(state,
                     model):
  return wrap_model_state(copy_dynamics(state["dynamics"], model),
                          state["static_forcing"],
                          copy_tracers(state["tracers"], model))


@partial(jit, static_argnames=["model"])
def wrap_tracers(moisture_species,
                 tracers,
                 model,
                 dry_air_species=None):
  tracer_struct = {"moisture_species": moisture_species,
                   "tracers": tracers}
  if dry_air_species is not None:
    tracer_struct["dry_air_species"] = dry_air_species
  if model in moist_mixing_ratio_models:
    tracer_struct["moist_mixing_ratio"] = 1.0
  else:
    tracer_struct["dry_mixing_ratio"] = 1.0
  return tracer_struct


@jit
def wrap_static_forcing(phi_surf,
                        grad_phi_surf,
                        coriolis_param,
                        nontrad_coriolis_param=None):
  static_forcing = {"phi_surf": phi_surf,
                    "grad_phi_surf": grad_phi_surf,
                    "coriolis_param": coriolis_param}
  if nontrad_coriolis_param is not None:
    static_forcing["nontrad_coriolis_param"] = nontrad_coriolis_param
  return static_forcing


def init_static_forcing(phi_surf,
                        h_grid,
                        physics_config,
                        dims,
                        model,
                        f_plane_center=jnp.pi / 4.0):
  grad_phi_surf_discont = horizontal_gradient(phi_surf, h_grid, a=physics_config["radius_earth"])
  grad_phi_surf = jnp.stack([project_scalar_global([grad_phi_surf_discont[:, :, :, 0]], h_grid, dims)[0],
                             project_scalar_global([grad_phi_surf_discont[:, :, :, 1]], h_grid, dims)[0]], axis=-1)
  if model in f_plane_models:
    coriolis_param = 2.0 * physics_config["period_earth"] * (jnp.sin(f_plane_center) *
                                                             jnp.ones_like(h_grid["physical_coords"][:, :, :, 0]))
  else:
    coriolis_param = 2.0 * physics_config["period_earth"] * jnp.sin(h_grid["physical_coords"][:, :, :, 0])
  if model in deep_atmosphere_models:
    nontrad_coriolis_param = 2.0 * physics_config["period_earth"] * jnp.cos(h_grid["physical_coords"][:, :, :, 0])
  else:
    nontrad_coriolis_param = None
  return wrap_static_forcing(phi_surf, grad_phi_surf, coriolis_param, nontrad_coriolis_param=nontrad_coriolis_param)


@partial(jit, static_argnames=["dims", "model"])
def project_dynamics(dynamics_in,
                     h_grid,
                     dims,
                     model):
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
  u_cont = project_scalar_3d(dynamics_in["u"][:, :, :, :, 0], h_grid, dims)
  v_cont = project_scalar_3d(dynamics_in["u"][:, :, :, :, 1], h_grid, dims)
  thermo_var_cont = project_scalar_3d(dynamics_in[thermodynamic_variable_names[model]][:, :, :, :], h_grid, dims)
  d_mass_cont = project_scalar_3d(dynamics_in["d_mass"][:, :, :, :], h_grid, dims)
  if model not in hydrostatic_models:
    w_i_cont = project_scalar_3d(dynamics_in["w_i"], h_grid, dims)
    phi_i_cont = project_scalar_3d(dynamics_in["phi_i"], h_grid, dims)
  else:
    phi_i_cont = None
    w_i_cont = None
  return wrap_dynamics(jnp.stack((u_cont, v_cont), axis=-1),
                       thermo_var_cont,
                       d_mass_cont,
                       model,
                       phi_i=phi_i_cont,
                       w_i=w_i_cont)


@partial(jit, static_argnames=["dims"])
def project_scalar_3d(variable,
                      h_grid,
                      dims):
  return project_scalar_global([variable],
                               h_grid,
                               dims,
                               two_d=False)[0]


@jit
def dynamics_to_surface_mass(state_in,
                             v_grid):
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
def remap_dynamics(dynamics_in,
                   static_forcing,
                   v_grid,
                   physics_config,
                   num_lev,
                   model):
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
  pi_surf = dynamics_to_surface_mass(dynamics_in, v_grid)
  d_mass_ref = surface_mass_to_d_mass(pi_surf,
                                      v_grid)
  d_mass = dynamics_in["d_mass"]
  u_model = dynamics_in["u"][:, :, :, :, 0] * d_mass
  v_model = dynamics_in["u"][:, :, :, :, 1] * d_mass
  if model in cam_se_models:
    thermo_model = dynamics_in["T"] * d_mass
  else:
    thermo_model = dynamics_in["theta_v_d_mass"]
  if model not in hydrostatic_models:
    p_mid = eval_midlevel_pressure(dynamics_in, v_grid)
    phi_ref = eval_balanced_geopotential(static_forcing["phi_surf"],
                                         p_mid,
                                         dynamics_in["theta_v_d_mass"],
                                         physics_config)
    phi_pert = dynamics_in["phi_i"] - phi_ref
    d_phi = interface_to_delta(phi_pert)
    dw = interface_to_delta(dynamics_in["w_i"])
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
    p_mid = surface_mass_to_midlevel_mass(pi_surf, v_grid)
    phi_ref_new = eval_balanced_geopotential(static_forcing["phi_surf"],
                                             p_mid,
                                             thermo_remap,
                                             physics_config)
    phi_i_remap = cumulative_sum(-Qdp_out[:, :, :, :, 3], jnp.zeros_like(static_forcing["phi_surf"])) + phi_ref_new
    w_i_surf = ((u_remap[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
                 u_remap[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
                phi_to_g(static_forcing["phi_surf"], physics_config, model))
    w_i_upper = flip(jnp.cumsum(-flip(Qdp[:, :, :, :, 4], -1), axis=-1), -1) + dynamics_in["w_i"][:, :, :, -1:]
    w_i_remap = jnp.concatenate((w_i_upper, w_i_surf[:, :, :, np.newaxis]), axis=-1)
  else:
    phi_i_remap = None
    w_i_remap = None
  return wrap_dynamics(u_remap,
                       thermo_remap,
                       d_mass_ref,
                       model,
                       phi_i=phi_i_remap,
                       w_i=w_i_remap)


@partial(jit, static_argnames=["model"])
def sum_dynamics(state1,
                 state2,
                 fold_coeff1,
                 fold_coeff2,
                 model):
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
  return wrap_dynamics(state1["u"] * fold_coeff1 + state2["u"] * fold_coeff2,
                       state1[thermo_var_name] * fold_coeff1 + state2[thermo_var_name] * fold_coeff2,
                       state1["d_mass"] * fold_coeff1 + state2["d_mass"] * fold_coeff2,
                       model,
                       phi_i=phi_i,
                       w_i=w_i)


@partial(jit, static_argnames=["model"])
def sum_dynamics_series(states,
                        coeffs,
                        model):
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
  state_out = sum_dynamics(states[0],
                           states[1],
                           coeffs[0],
                           coeffs[1], model)
  for coeff_idx in range(2, len(states)):
    state_out = sum_dynamics(state_out,
                             states[coeff_idx],
                             1.0,
                             coeffs[coeff_idx], model)
  return state_out


def check_dynamics_nan(dynamics,
                       model):
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
  is_nan = False
  fields = ["u", thermodynamic_variable_names[model], "d_mass"]
  if model not in hydrostatic_models:
    fields += ["w_i", "phi_i"]
  for field in fields:
    is_nan = is_nan or jnp.any(jnp.isnan(dynamics[field]))
  is_nan = int(is_nan)
  return global_sum(is_nan) > 0


def check_tracers_nan(tracers,
                      model):
  is_nan = False
  for field_name in tracers["moisture_species"].keys():
    is_nan = is_nan or jnp.any(jnp.isnan(tracers["moisture_species"][field_name]))
  for field_name in tracers["tracers"].keys():
    is_nan = is_nan or jnp.any(jnp.isnan(tracers["tracers"][field_name]))
  if model in cam_se_models:
    for field_name in tracers["dry_air_species"].keys():
      is_nan = is_nan or jnp.any(jnp.isnan(tracers["dry_air_species"][field_name]))
  is_nan = int(is_nan)
  return global_sum(is_nan) > 0
