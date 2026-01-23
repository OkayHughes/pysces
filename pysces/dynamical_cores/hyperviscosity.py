from ..config import vmap_1d_apply, jit, jnp, device_wrapper, np
from ..operations_2d.operators import horizontal_weak_vector_laplacian, horizontal_weak_laplacian
from ..operations_2d.tensor_hyperviscosity import quasi_uniform_hypervisc_coeff, variable_resolution_hypervisc_coeff
from ..horizontal_grid import get_global_grid_deformation_metrics
from .model_state import wrap_dynamics_struct, project_dynamics_state
from ..config import jnp, jit, np, device_wrapper
from .utils_3d import get_delta, interface_to_model
from .homme.thermodynamics import get_balanced_phi
from .mass_coordinate import mass_from_coordinate_interface
from functools import partial
from ..model_info import hydrostatic_models, thermodynamic_variable_names, homme_models, cam_se_models


@jit
def scalar_harmonic_3d(scalar, h_grid, physics_config):
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
  def lap_wk_onearg(scalar):
      return horizontal_weak_laplacian(scalar, h_grid, a=physics_config["radius_earth"])

  del2 = vmap_1d_apply(lap_wk_onearg, scalar, -1, -1)
  return del2


@jit
def vector_harmonic_3d(vector, h_grid, physics_config, nu_div_factor):
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
  def vec_lap_wk_onearg(vector):
      return horizontal_weak_vector_laplacian(vector, h_grid, a=physics_config["radius_earth"],
                                              nu_div_fact=nu_div_factor)

  del2 = vmap_1d_apply(vec_lap_wk_onearg, vector, -2, -2)
  return del2


@partial(jit, static_argnames=["apply_nu", "model"])
def calc_hypervis_harmonic(dynamics, h_grid, physics_config, diffusion_config, model, apply_nu=True):
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

  if apply_nu:
    nu_default = diffusion_config["nu"]
    nu_phi = diffusion_config["nu_phi"]
    nu_d_mass = diffusion_config["nu_d_mass"]
    nu_div_factor = diffusion_config["nu_div_factor"]
  else:
    nu_default = 1.0
    nu_phi = 1.0
    nu_d_mass = 1.0
    nu_div_factor = 1.0

  hyperdiff_u = vector_harmonic_3d(dynamics["u"],
                                   h_grid, physics_config, nu_div_factor)
  hyperdiff_d_mass = scalar_harmonic_3d(dynamics["d_mass"], h_grid, physics_config)
  if model not in hydrostatic_models:
    hyperdiff_phi_i = nu_phi * scalar_harmonic_3d(dynamics["phi_i"],
                                         h_grid, physics_config)
    hyperdiff_w_i = nu_default * scalar_harmonic_3d(dynamics["w_i"],
                                       h_grid, physics_config)
  else:
    hyperdiff_phi_i = None
    hyperdiff_w_i = None

  if model in homme_models:
    hyperdiff_thermo = scalar_harmonic_3d(dynamics["theta_v_d_mass"] / dynamics["d_mass"],
                                              h_grid, physics_config) * dynamics["d_mass"]
  else:
    hyperdiff_thermo = scalar_harmonic_3d(dynamics["T"], h_grid, physics_config)
  hypervis_tend = wrap_dynamics_struct(nu_default * hyperdiff_u,
                                        nu_default * hyperdiff_thermo,
                                        nu_d_mass * hyperdiff_d_mass,
                                        model,
                                        phi_i=hyperdiff_phi_i,
                                        w_i=hyperdiff_w_i)
  return hypervis_tend


@partial(jit, static_argnames=["dims", "model"])
def sponge_layer(dynamics, dt, h_grid, physics_config, diffusion_config, dims, model):
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
  nu_top = diffusion_config["nu_top"]
  nu_ramp = nu_top * diffusion_config["nu_ramp"]
  n_sponge = nu_ramp.size
  if model not in hydrostatic_models:
    hyperdiff_phi_i = nu_ramp * scalar_harmonic_3d(dynamics["phi_i"][:, :, :, :n_sponge],
                                                   h_grid, physics_config)
    hyperdiff_w_i = nu_ramp * scalar_harmonic_3d(dynamics["w_i"][:, :, :, :n_sponge],
                                                 h_grid, physics_config)
  else:
    hyperdiff_phi_i = None
    hyperdiff_w_i = None
  thermo_var_name = thermodynamic_variable_names[model]
  hyperdiff_thermo = scalar_harmonic_3d(dynamics[thermo_var_name][:, :, :, :n_sponge],
                                        h_grid, physics_config)
  hyperdiff_thermo *= nu_ramp
  hyperdiff_d_mass = scalar_harmonic_3d(dynamics["d_mass"][:, :, :, :n_sponge], h_grid, physics_config)
  hyperdiff_d_mass *= nu_ramp
  hyperdiff_u = vector_harmonic_3d(dynamics["u"][:, :, :, :n_sponge, :],
                                   h_grid, physics_config, 1.0)
  hyperdiff_u *= nu_ramp[:, :, :, :, np.newaxis]
  hyperdiff_state = wrap_dynamics_struct(hyperdiff_u,
                                         hyperdiff_thermo,
                                         hyperdiff_d_mass,
                                         model,
                                         phi_i=hyperdiff_phi_i,
                                         w_i=hyperdiff_w_i)
  hyperdiff_state = project_dynamics_state(hyperdiff_state,
                                           h_grid,
                                           dims,
                                           model,
                                           scaled=False)

  u_out = jnp.concatenate((dt * hyperdiff_state["u"] + dynamics["u"][:, :, :, :n_sponge, :],
                           dynamics["u"][:, :, :, n_sponge:, :]), axis=-2)
  thermo_out = jnp.concatenate((dt * hyperdiff_state[thermo_var_name] + dynamics[thermo_var_name][:, :, :, :n_sponge],
                                dynamics[thermo_var_name][:, :, :, n_sponge:]), axis=-1)
  d_mass_out = jnp.concatenate((dt * hyperdiff_state["d_mass"] + dynamics["d_mass"][:, :, :, :n_sponge],
                             dynamics["d_mass"][:, :, :, n_sponge:]), axis=-1)
  if model not in hydrostatic_models:
    phi_i_out = jnp.concatenate((dt * hyperdiff_state["phi_i"] + dynamics["phi_i"][:, :, :, :n_sponge],
                                 dynamics["phi_i"][:, :, :, n_sponge:]), axis=-1)
    w_i_out = jnp.concatenate((dt * hyperdiff_state["phi_i"] + dynamics["w_i"][:, :, :, :n_sponge],
                               dynamics["w_i"][:, :, :, n_sponge:]), axis=-1)
  else:
    phi_i_out = None
    w_i_out = None

  if model not in hydrostatic_models:
    phi_i_out = jnp.concatenate((dynamics["phi_i"][:, :, :, :n_sponge],
                                 dynamics["phi_i"][:, :, :, n_sponge:]), axis=-1)
    w_i_out = jnp.concatenate((dynamics["w_i"][:, :, :, :n_sponge],
                               dynamics["w_i"][:, :, :, n_sponge:]), axis=-1)
  else:
    phi_i_out = None
    w_i_out = None
  struct = wrap_dynamics_struct(u_out,
                                thermo_out,
                                d_mass_out,
                                phi_i=phi_i_out,
                                w_i=w_i_out)
  return struct

@partial(jit, static_argnames=["n_sponge"])
def get_nu_ramp(v_grid, n_sponge):
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
  pressure_ratio = ((v_grid["hybrid_a_i"][0] + v_grid["hybrid_b_i"][0]) /
                    (v_grid["hybrid_a_i"][:n_sponge] + v_grid["hybrid_b_i"][:n_sponge]))
  nu_ramp = jnp.minimum(device_wrapper(8.0),
                        (16.0 * pressure_ratio**2 /
                         (pressure_ratio**2 + 1.0))[np.newaxis, np.newaxis, np.newaxis, :])
  return nu_ramp


def init_hypervis_config_const(ne, config,
                               v_grid,
                               nu_top=2.5e5,
                               nu_base=-1.0,
                               nu_phi=-1.0,
                               nu_d_mass=-1.0,
                               nu_div_factor=2.5,
                               n_sponge=5,
                               T_ref=288.0,
                               T_ref_lapse=0.0065):
  nu = quasi_uniform_hypervisc_coeff(ne, config) if nu_base <= 0 else nu_base
  nu_phi = nu_base if nu_phi < 0 else nu_phi
  nu_d_mass= nu_base if nu_d_mass < 0 else nu_d_mass
  nu_ramp = get_nu_ramp(v_grid, n_sponge)
  diffusion_config = {"constant_hypervis": 1.0,
                      "nu": device_wrapper(nu),
                      "nu_phi": device_wrapper(nu_phi),
                      "nu_d_mass": device_wrapper(nu_d_mass),
                      "nu_div_factor": device_wrapper(nu_div_factor),
                      "reference_profiles": {"T_ref": device_wrapper(T_ref),
                                             "T_ref_lapse": device_wrapper(T_ref_lapse)}}

  if n_sponge > 0:
    diffusion_config["sponge_layer"] = 1.0
    diffusion_config["nu_top"] = device_wrapper(nu_top)
    diffusion_config["nu_ramp"] = device_wrapper(nu_ramp)
  return diffusion_config

def init_hypervis_config_stub():
  return {"disable_diffusion": 1.0}

def init_hypervis_config_tensor(h_grid, v_grid, dims, config,
                                nu_top=2.5e5,
                                ad_hoc_scale=1.0,
                                n_sponge=5):
  nu_ramp = get_nu_ramp(v_grid, n_sponge)
  radius_earth = config["radius_earth"]
  _, max_min_dx, _ = get_global_grid_deformation_metrics(h_grid, dims)
  print(radius_earth)
  print(max_min_dx)
  nu_tens = variable_resolution_hypervisc_coeff(radius_earth * max_min_dx,
                                                h_grid["hypervis_scaling"],
                                                dims["npt"])
  nu = device_wrapper(ad_hoc_scale * nu_tens)
  diffusion_config = {"tensor_hypervis": 1.0,
                      "nu": nu,
                      "nu_phi": nu,
                      "nu_d_mass": nu}
  if n_sponge > 0:
    diffusion_config["do_sponge_layer"] = 1.0
    diffusion_config["nu_top"] = device_wrapper(nu_top)
    diffusion_config["nu_ramp"] = device_wrapper(nu_ramp)
  return diffusion_config


@partial(jit, static_argnames=["model", "dims"])
def hypervis_terms(dynamics, static_forcing, h_grid, v_grid, dims, physics_config, diffusion_config, model):
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
  ref_state = get_ref_states(static_forcing["phi_surf"], v_grid, physics_config, diffusion_config, model)
  if model not in hydrostatic_models:
    phi_i_pert = dynamics["phi_i"] - ref_state["phi_i"]
    w_i = dynamics["w_i"]
  else:
    phi_i_pert = None
    w_i = None
  
  if model in cam_se_models:
    thermo_var = dynamics["T"]
    thermo_var -= ref_state["T"]
  else:
    thermo_var = dynamics["theta_v_d_mass"] / dynamics["d_mass"]
    thermo_var -= ref_state["theta_v"]


  hypervis_state = wrap_dynamics_struct(dynamics["u"],
                                        thermo_var,
                                        dynamics["d_mass"],
                                        model,
                                        phi_i=phi_i_pert,
                                        w_i=w_i)
  for apply_nu in [True, False]:
    hypervis_state = calc_hypervis_harmonic(hypervis_state,
                                            h_grid,
                                            physics_config,
                                            diffusion_config,
                                            model,
                                            apply_nu=apply_nu)
    
    hypervis_state = project_dynamics_state(hypervis_state, h_grid, dims, model, scaled=False)
  return wrap_dynamics_struct()


@jit
def get_ref_states(phi_surf, v_grid, physics_config, diffusion_config, model):
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
  # could eventually only be called once.
  # due to low cost, if we end up going the "vmap over nelem" route,
  # then this should probably be recomputed from element-local (and ideally SM-local) phi_surf
  reference_params = diffusion_config["reference_profiles"]
  dummy_thermo = reference_params["dummy_thermodynamics"]
  ps_ref = v_grid["reference_surface_mass"] * jnp.exp(-phi_surf / (dummy_thermo["Rgas"] * reference_params["T_ref"]))
  pressure_int = mass_from_coordinate_interface(ps_ref, v_grid)
  d_mass_ref = get_delta(pressure_int)
  p_mid = interface_to_model(pressure_int)
  exner = (p_mid / dummy_thermo["p0"])**(dummy_thermo["Rgas"] / dummy_thermo["cp"])
  T1 = reference_params["T_ref_lapse"] * reference_params["T_ref"] * dummy_thermo["cp"] / physics_config["gravity"]
  T0 = reference_params["T_ref"] - T1
  theta_ref = T0 + T0 * (1 - exner) + T1
  if model in cam_se_models:
    thermo_var_name = "T"
    thermo_profile = theta_ref * exner
  elif model in homme_models:
    thermo_var_name = "theta_v"
    thermo_profile = theta_ref
  ref_profiles = {"d_mass": d_mass_ref,
                  thermo_var_name: thermo_profile}
  if model not in hydrostatic_models:
    ref_profiles["phi_i"] = get_balanced_phi(phi_surf, p_mid, theta_ref * d_mass_ref, physics_config)
  return ref_profiles
