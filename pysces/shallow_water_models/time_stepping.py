from ..config import jit, DEBUG
from .explicit_terms import calc_rhs
from .model_state import project_state, advance_state
from .hyperviscosity import calc_hypervis
from ..time_step import time_step_options, stability_info
from ..operations_2d.tensor_hyperviscosity import get_cfl
from frozendict import frozendict
from functools import partial


@partial(jit, static_argnames=["dims"])
def advance_step_euler(state_in, dt, grid, config, dims):
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
  state_tend = calc_rhs(state_in, grid, config)
  state_tend_c0 = project_state(state_tend, grid, dims)
  return advance_state([state_in, state_tend_c0], [1.0, dt])


@partial(jit, static_argnames=["dims"])
def advance_step_ssprk3(state0, dt, grid, config, dims):
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
  tend = calc_rhs(state0, grid, config)
  tend_c0 = project_state(tend, grid, dims)
  state1 = advance_state([state0, tend_c0], [1.0, dt])
  tend = calc_rhs(state1, grid, config)
  tend_c0 = project_state(tend, grid, dims)
  state2 = advance_state([state0, state1, tend_c0], [3.0 / 4.0,
                                                     1.0 / 4.0,
                                                     1.0 / 4.0 * dt])
  tend = calc_rhs(state2, grid, config)
  tend_c0 = project_state(tend, grid, dims)
  return advance_state([state0, state2, tend_c0], [1.0 / 3.0,
                                                   2.0 / 3.0,
                                                   2.0 / 3.0 * dt])


@partial(jit, static_argnames=["dims", "substeps"])
def advance_hypervis_euler(state_in, dt, grid, config, dims, substeps=1):
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
  next_step = state_in
  for k in range(substeps):
    hvis_tend = calc_hypervis(next_step, grid, config, dims)
    next_step = advance_state([next_step, hvis_tend], [1.0, dt / substeps])
  return next_step


def get_timestep_config(dt_coupling,
                        h_grid,
                        v_grid,
                        physics_config,
                        diffusion_config,
                        tracer_tstep_type=time_step_options.RK2,
                        hypervis_tstep_type=time_step_options.Euler,
                        dynamics_tstep_type=time_step_options.RK3_5STAGE,
                        sponge_tstep_type=time_step_options.Euler,
                        tracer_steps_per_coupling_interval=-1,
                        dyn_steps_per_tracer=-1,
                        hypervis_steps_per_dyn=-1,
                        sponge_steps_per_dyn=-1,
                        sphere=True):
  cfl_info = get_cfl(h_grid, v_grid, physics_config, diffusion_config, sphere=sphere)
  tracer_S = stability_info[tracer_tstep_type]
  hypervisc_S = stability_info[hypervis_tstep_type]
  dynamics_S = stability_info[dynamics_tstep_type]
  sponge_S = stability_info[sponge_tstep_type]
  #rkssp_euler_stability = cfl_info["dt_rkssp_euler"]
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
  max_dt_dynamics = dynamics_S * dt_gravity_wave
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
    print(f"Stability: nu_dpi  hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_scalar}s")
    print(f"Stability: nu_vor hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_vort}s")
    print(f"Stability: nu_div hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_div}s")
    print(f"scaled nu_top viscosity CFL: dt = {dt_sponge}s < {max_dt_sponge}s")

  return frozendict(tracer_advection=frozendict(step_type=tracer_tstep_type,
                                                dt=dt_tracer),
                    dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics),
                    hyperviscosity=frozendict(step_type=hypervis_tstep_type,
                                              dt=dt_hypervis),
                    sponge=frozendict(step_type=sponge_tstep_type,
                                      dt=dt_sponge),
                    tracer_subcycle=tracer_subcycle,
                    dynamics_subcycle=dynamics_subcycle,
                    hypervisc_subcycle=hypervisc_subcycle,
                    sponge_subcycle=sponge_subcycle)