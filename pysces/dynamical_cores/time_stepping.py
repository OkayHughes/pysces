from ..config import jit, jnp, DEBUG
from .model_state import project_dynamics, sum_dynamics_series, sum_consistency_struct
from .homme.explicit_terms import eval_explicit_tendency as eval_explicit_tendency_homme
from .cam_se.explicit_terms import eval_explicit_tendency as eval_explicit_tendency_se
from .homme.explicit_terms import correct_state
from .hyperviscosity import eval_hypervis_terms, advance_sponge_layer
from ..horizontal_grid import eval_cfl
from ..time_step import time_step_options, stability_info
from functools import partial
from frozendict import frozendict
from .physics_dynamics_coupling import coupling_types
from ..model_info import cam_se_models, homme_models


@partial(jit, static_argnames=["dims", "model"])
def dynamics_tendency(dynamics,
                      static_forcing,
                      h_grid,
                      v_grid,
                      physics_config,
                      dims,
                      model,
                      moisture_species=None,
                      dry_air_species=None):
  if model in cam_se_models:
    dynamics_tend, tracer_consist = eval_explicit_tendency_se(dynamics,
                                                              static_forcing,
                                                              moisture_species,
                                                              dry_air_species,
                                                              h_grid,
                                                              v_grid,
                                                              physics_config,
                                                              model)
  elif model in homme_models:
    dynamics_tend, tracer_consist = eval_explicit_tendency_homme(dynamics,
                                                                 static_forcing,
                                                                 h_grid,
                                                                 v_grid,
                                                                 physics_config,
                                                                 model)
  dynamics_tend_c0 = project_dynamics(dynamics_tend, h_grid, dims, model)
  return dynamics_tend_c0, tracer_consist


@partial(jit, static_argnames=["model"])
def enforce_conservation(dynamics,
                         static_forcing,
                         dt,
                         physics_config,
                         model):
  if model in cam_se_models:
    dynamics_conserve = dynamics
  else:
    dynamics_conserve = correct_state(dynamics, static_forcing, dt, physics_config, model)
  dynamics_conserve = dynamics
  return dynamics_conserve


def eval_cfl_3d(h_grid,
                physics_config,
                diffusion_config,
                dims,
                model):
  cfl_info, grid_info = eval_cfl(h_grid, physics_config["radius_earth"], diffusion_config, dims, model)
  max_norm_jac_inv = grid_info["max_norm_jac_inv"]

  if "sponge_layer" in diffusion_config.keys():
    nu_top_max = jnp.max(diffusion_config["nu_ramp"]) * diffusion_config["nu_top"]
    sponge_layer_stab = 1.0 / (nu_top_max * ((grid_info["scale_inv"] * max_norm_jac_inv)**2) * grid_info["lambda_vis"])
    cfl_info["dt_sponge_layer"] = sponge_layer_stab
  else:
    cfl_info["dt_sponge_layer"] = 1e6
  return cfl_info


def init_timestep_config(dt_coupling,
                         h_grid,
                         physics_config,
                         diffusion_config,
                         dims,
                         model,
                         tracer_tstep_type=time_step_options.RK2,
                         hypervis_tstep_type=time_step_options.Euler,
                         dynamics_tstep_type=time_step_options.RK3_5STAGE,
                         sponge_tstep_type=time_step_options.Euler,
                         tracer_steps_per_coupling_interval=-1,
                         dyn_steps_per_tracer=-1,
                         hypervis_steps_per_dyn=-1,
                         sponge_steps_per_dyn=-1,
                         physics_dynamics_coupling=coupling_types.none,
                         print_cfl=False):
  cfl_info = eval_cfl_3d(h_grid, physics_config, diffusion_config, dims, model)
  tracer_S = stability_info[tracer_tstep_type]
  hypervisc_S = stability_info[hypervis_tstep_type]
  dynamics_S = stability_info[dynamics_tstep_type]
  sponge_S = stability_info[sponge_tstep_type]
  # rkssp_euler_stability = cfl_info["dt_rkssp_euler"]
  dt_rk2_tracer = cfl_info["dt_rk2_tracer"]
  dt_gravity_wave = cfl_info["dt_gravity_wave"]
  dt_hypervis_scalar = cfl_info["dt_hypervis_scalar"]
  dt_hypervis_vort = cfl_info["dt_hypervis_vort"]
  dt_hypervis_div = cfl_info["dt_hypervis_div"]
  dt_sponge_layer = cfl_info["dt_sponge_layer"]

  # determine q_split
  max_dt_scalar = tracer_S * dt_rk2_tracer
  # we are assuming remap and tracer advection are done at the
  # same frequency!
  tracer_subcycle = max(int(dt_coupling / max_dt_scalar) + 1, tracer_steps_per_coupling_interval)
  dt_tracer = dt_coupling / tracer_subcycle

  # determine n_split
  max_dt_dynamics = dynamics_S * dt_gravity_wave / 2.0
  dynamics_subcycle = max(int(dt_tracer / max_dt_dynamics) + 1, dyn_steps_per_tracer)
  dt_dynamics = dt_tracer / dynamics_subcycle

  # determine hv_split
  max_dt_hypervis_scalar = hypervisc_S * dt_hypervis_scalar
  max_dt_hypervis_vort = hypervisc_S * dt_hypervis_vort
  max_dt_hypervis_div = hypervisc_S * dt_hypervis_div
  max_dt_hypervis = min([max_dt_hypervis_scalar,
                         max_dt_hypervis_vort,
                         max_dt_hypervis_div])
  hypervisc_subcycle = max(int(dt_dynamics / max_dt_hypervis) + 1, hypervis_steps_per_dyn)
  dt_hypervis = dt_dynamics / hypervisc_subcycle

  # determine sponge_split
  max_dt_sponge = sponge_S * dt_sponge_layer
  sponge_subcycle = max(int(dt_dynamics / max_dt_sponge) + 1, sponge_steps_per_dyn)
  dt_sponge = dt_dynamics / sponge_subcycle
  if DEBUG:
    print("CFL estimates:")
    # print(f"SSP preservation (120m/s) RKSSP euler step dt  < S * {rkssp_euler_stability}s")
    print(f"Stability: advective (120m/s)   dt_tracer = {dt_tracer}s <  {max_dt_scalar}s")
    print(f"Stability: gravity wave(342m/s)   dt_dyn = {dt_dynamics}s  < {max_dt_dynamics}s")
    #  dt < S  1 / nu * norm_jac_inv_hypervis
    print(f"Stability: nu_d_mass  hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_scalar}s")
    print(f"Stability: nu_vor hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_vort}s")
    print(f"Stability: nu_div hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_div}s")
    print(f"scaled nu_top viscosity CFL: dt = {dt_sponge}s < {max_dt_sponge}s")

  return frozendict(tracer_advection=frozendict(step_type=tracer_tstep_type,
                                                dt=dt_tracer),
                    dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics,
                                        tracer_consistency_frac=1.0 / tracer_subcycle),
                    hyperviscosity=frozendict(step_type=hypervis_tstep_type,
                                              dt=dt_hypervis,
                                              tracer_consistency_frac=1.0 / (tracer_subcycle * dynamics_subcycle)),
                    sponge=frozendict(step_type=sponge_tstep_type,
                                      dt=dt_sponge),
                    physics_dt=dt_coupling,
                    tracer_subcycle=tracer_subcycle,
                    dynamics_subcycle=dynamics_subcycle,
                    hypervis_subcycle=hypervisc_subcycle,
                    sponge_subcycle=sponge_subcycle,
                    physics_dynamics_coupling=physics_dynamics_coupling)


@partial(jit, static_argnames=["dims", "model", "timestep_config"])
def advance_dynamics_euler(dynamics_in,
                           static_forcing,
                           h_grid,
                           v_grid,
                           physics_config,
                           timestep_config,
                           dims,
                           model,
                           moisture_species=None,
                           dry_air_species=None):
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
  dt = timestep_config["dynamics"]["dt"]
  dynamics_tend_cont, tracer_consist = dynamics_tendency(dynamics_in,
                                                         static_forcing,
                                                         h_grid,
                                                         v_grid,
                                                         physics_config,
                                                         dims,
                                                         model,
                                                         moisture_species=moisture_species,
                                                         dry_air_species=dry_air_species)
  dynamics_out_discont = sum_dynamics_series([dynamics_in, dynamics_tend_cont],
                                             [1.0, dt],
                                             model)
  dynamics_out_cont = enforce_conservation(dynamics_out_discont,
                                           static_forcing,
                                           dt,
                                           physics_config,
                                           model)
  tracer_consist = sum_consistency_struct(tracer_consist,
                                          tracer_consist,
                                          0.0,
                                          1.0 / timestep_config["dynamics_subcycle"])
  return dynamics_out_cont, tracer_consist


@partial(jit, static_argnames=["dims", "model", "timestep_config"])
def advance_hypervis_euler(dynamics,
                           static_forcing,
                           h_grid,
                           v_grid,
                           physics_config,
                           diffusion_config,
                           timestep_config,
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
  state_out = dynamics
  tracer_consist_frac = 1.0 / (timestep_config["dynamics_subcycle"] * timestep_config["hypervis_subcycle"])
  for subcycle_idx in range(timestep_config["hypervis_subcycle"]):
    hypervis_rhs, tracer_consist = eval_hypervis_terms(state_out,
                                                       static_forcing,
                                                       h_grid,
                                                       v_grid,
                                                       dims,
                                                       physics_config,
                                                       diffusion_config,
                                                       model)
    if subcycle_idx > 0:
      tracer_consist_total = sum_consistency_struct(tracer_consist_total,
                                                    tracer_consist,
                                                    1.0,
                                                    tracer_consist_frac)
    else:
      tracer_consist_total = sum_consistency_struct(tracer_consist,
                                                    tracer_consist,
                                                    tracer_consist_frac,
                                                    0.0)
    state_out = sum_dynamics_series([state_out, hypervis_rhs], [1.0, timestep_config["hyperviscosity"]["dt"]], model)
  # Todo: figure out lower boundary correction.
  return state_out, tracer_consist_total


@partial(jit, static_argnames=["dims", "model", "timestep_config"])
def advance_sponge_euler(dynamics,
                         h_grid,
                         physics_config,
                         diffusion_config,
                         timestep_config,
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
  dynamics_out = dynamics
  for _ in range(timestep_config["sponge_subcycle"]):
    dynamics_out = advance_sponge_layer(dynamics_out,
                                        timestep_config["sponge"]["dt"],
                                        h_grid,
                                        physics_config,
                                        diffusion_config,
                                        dims,
                                        model)
  return dynamics_out


@partial(jit, static_argnames=["dims", "model", "timestep_config"])
def advance_dynamics_ullrich_5stage(dynamics_in,
                                    static_forcing,
                                    h_grid,
                                    v_grid,
                                    physics_config,
                                    timestep_config,
                                    dims,
                                    model,
                                    moisture_species=None,
                                    dry_air_species=None):
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
  dt = timestep_config["dynamics"]["dt"]
  tracer_consist_frac = 1.0 / timestep_config["dynamics_subcycle"]
  dynamics_tend, tracer_consist_0 = dynamics_tendency(dynamics_in,
                                                      static_forcing,
                                                      h_grid,
                                                      v_grid,
                                                      physics_config,
                                                      dims,
                                                      model,
                                                      moisture_species=moisture_species,
                                                      dry_air_species=dry_air_species)
  dynamics_keep = sum_dynamics_series([dynamics_in, dynamics_tend], [1.0, dt / 5.0], model)
  dynamics_keep = enforce_conservation(dynamics_keep,
                                       static_forcing,
                                       dt / 5.0,
                                       physics_config,
                                       model)

  dynamics_tend, _ = dynamics_tendency(dynamics_keep,
                                       static_forcing,
                                       h_grid,
                                       v_grid,
                                       physics_config,
                                       dims,
                                       model,
                                       moisture_species=moisture_species,
                                       dry_air_species=dry_air_species)
  dynamics_tmp = sum_dynamics_series([dynamics_in, dynamics_tend],
                                     [1.0, dt / 5.0],
                                     model)
  dynamics_tmp = enforce_conservation(dynamics_tmp,
                                      static_forcing,
                                      dt / 5.0,
                                      physics_config,
                                      model)

  dynamics_tend, _ = dynamics_tendency(dynamics_tmp,
                                       static_forcing,
                                       h_grid,
                                       v_grid,
                                       physics_config,
                                       dims,
                                       model,
                                       moisture_species=moisture_species,
                                       dry_air_species=dry_air_species)
  dynamics_tmp = sum_dynamics_series([dynamics_in, dynamics_tend],
                                     [1.0, dt / 3.0],
                                     model)
  dynamics_tmp = enforce_conservation(dynamics_tmp,
                                      static_forcing,
                                      dt / 3.0,
                                      physics_config,
                                      model)

  dynamics_tend, _ = dynamics_tendency(dynamics_tmp,
                                       static_forcing,
                                       h_grid,
                                       v_grid,
                                       physics_config,
                                       dims,
                                       model,
                                       moisture_species=moisture_species,
                                       dry_air_species=dry_air_species)
  dynamics_tmp = sum_dynamics_series([dynamics_in, dynamics_tend],
                                     [1.0, 2.0 * dt / 3.0],
                                     model)
  dynamics_tmp = enforce_conservation(dynamics_tmp,
                                      static_forcing,
                                      2.0 * dt / 3.0,
                                      physics_config,
                                      model)

  dynamics_tend, tracer_consist_1 = dynamics_tendency(dynamics_tmp,
                                                      static_forcing,
                                                      h_grid,
                                                      v_grid,
                                                      physics_config,
                                                      dims,
                                                      model,
                                                      moisture_species=moisture_species,
                                                      dry_air_species=dry_air_species)
  final_state = sum_dynamics_series([dynamics_in,
                                     dynamics_keep,
                                     dynamics_tend],
                                    [-1.0 / 4.0,
                                     5.0 / 4.0,
                                     3.0 * dt / 4.0],
                                    model)
  final_state = enforce_conservation(final_state,
                                     static_forcing,
                                     2.0 * dt / 3.0,
                                     physics_config,
                                     model)
  tracer_consist_total = sum_consistency_struct(tracer_consist_0,
                                                tracer_consist_1,
                                                1.0 / 4.0 * tracer_consist_frac,
                                                3.0 / 4.0 * tracer_consist_frac)
  return final_state, tracer_consist_total
