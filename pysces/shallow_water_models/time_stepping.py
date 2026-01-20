from ..config import jit, DEBUG
from .explicit_terms import calc_rhs
from .model_state import project_state, advance_state
from .hyperviscosity import calc_hypervis_quasi_uniform, calc_hypervis_variable_resolution
from ..time_step import time_step_options, stability_info
from ..operations_2d.se_grid import get_cfl
from frozendict import frozendict
from functools import partial


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_step_euler(state_in, grid, physics_config, timestep_config, dims):
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

  state_tend = calc_rhs(state_in, grid, physics_config)
  state_tend_c0 = project_state(state_tend, grid, dims)
  return advance_state([state_in, state_tend_c0], [1.0, timestep_config["dynamics"]["dt"]])


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_step_ssprk3(state0, grid, physics_config, timestep_config, dims):
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
  tend = calc_rhs(state0, grid, physics_config)
  tend_c0 = project_state(tend, grid, dims)
  state1 = advance_state([state0, tend_c0], [1.0, dt])
  tend = calc_rhs(state1, grid, physics_config)
  tend_c0 = project_state(tend, grid, dims)
  state2 = advance_state([state0, state1, tend_c0], [3.0 / 4.0,
                                                     1.0 / 4.0,
                                                     1.0 / 4.0 * dt])
  tend = calc_rhs(state2, grid, physics_config)
  tend_c0 = project_state(tend, grid, dims)
  return advance_state([state0, state2, tend_c0], [1.0 / 3.0,
                                                   2.0 / 3.0,
                                                   2.0 / 3.0 * dt])


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_hypervis_euler(state_in, grid, physics_config, diffusion_config, timestep_config, dims):
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
  for k in range(timestep_config["hypervis_subcycle"]):
    if "tensor_hypervis" in diffusion_config.keys():
      hvis_tend = calc_hypervis_variable_resolution(next_step, grid, physics_config, diffusion_config, dims)
    else:
      hvis_tend = calc_hypervis_quasi_uniform(next_step, grid, physics_config, diffusion_config, dims)
    next_step = advance_state([next_step, hvis_tend], [1.0, -timestep_config["hyperviscosity"]["dt"]])
  return next_step


def get_timestep_config(dt_coupling,
                        h_grid,
                        dims,
                        physics_config,
                        diffusion_config,
                        hypervis_tstep_type=time_step_options.Euler,
                        dynamics_tstep_type=time_step_options.SSPRK3,
                        dyn_steps_per_coupling=-1,
                        hypervis_steps_per_dyn=-1,
                        sphere=True):
  cfl_info, _ = get_cfl(h_grid, physics_config["radius_earth"], diffusion_config, dims, sphere=sphere)
  hypervisc_S = stability_info[hypervis_tstep_type]
  dynamics_S = stability_info[dynamics_tstep_type]
  dt_rkssp_stability = cfl_info["dt_rkssp_euler"]
  dt_hypervis_scalar = cfl_info["dt_hypervis_scalar"]
  dt_hypervis_vort = cfl_info["dt_hypervis_vort"]
  dt_hypervis_div = cfl_info["dt_hypervis_div"]

  # determine n_split
  max_dt_dynamics = dynamics_S * dt_rkssp_stability
  dynamics_subcycle = max(int(dt_coupling / max_dt_dynamics) + 1, dyn_steps_per_coupling)
  dt_dynamics = dt_coupling / dynamics_subcycle

  # determine hv_split
  max_dt_hypervis_scalar = hypervisc_S * dt_hypervis_scalar
  max_dt_hypervis_vort = hypervisc_S * dt_hypervis_vort
  max_dt_hypervis_div = hypervisc_S * dt_hypervis_div
  # calculate tightest constraint on HV stability
  max_dt_hypervis = min([max_dt_hypervis_scalar,
                         max_dt_hypervis_vort,
                         max_dt_hypervis_div])
  hypervisc_subcycle = max(int(dt_dynamics / max_dt_hypervis) + 1, hypervis_steps_per_dyn)
  dt_hypervis = dt_dynamics / hypervisc_subcycle

  if DEBUG:
    print("CFL estimates:")
    # print(f"SSP preservation (120m/s) RKSSP euler step dt  < S * {rkssp_euler_stability}s")
    print(f"Stability: RKSSP preservation (120m/s) dt_dyn = {dt_dynamics}s  < {max_dt_dynamics}s")
    #  dt < S  1 / nu * norm_jac_inv_hypervis
    print(f"Stability: nu_dpi  hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_scalar}s")
    print(f"Stability: nu_vor hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_vort}s")
    print(f"Stability: nu_div hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_div}s")

  return frozendict(dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics),
                    hyperviscosity=frozendict(step_type=hypervis_tstep_type,
                                              dt=dt_hypervis),
                    dt_coupling=dt_coupling,
                    dynamics_subcycle=dynamics_subcycle,
                    hypervis_subcycle=hypervisc_subcycle)
