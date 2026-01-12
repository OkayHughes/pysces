from ...config import jit, jnp, DEBUG
from .model_state import wrap_model_struct, project_model_state
from .explicit_terms import explicit_tendency, correct_state
from .theta_hyperviscosity import hypervis_terms, sponge_layer
from ..hyperviscosity import get_global_grid_defomation_metrics
from functools import partial
from enum import Enum
from frozendict import frozendict

time_step_options = Enum('time_step', [("Euler", 1),
                                       ("RK3_5STAGE", 2),
                                       ("RK2", 3)])

stability_info = {time_step_options.RK2: 2.0,
                  time_step_options.Euler: 2.0,
                  time_step_options.RK3_5STAGE: 3.87}

@jit
def rfold_state(state1, state2, fold_coeff1, fold_coeff2):
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
  return wrap_model_struct(state1["u"] * fold_coeff1 + state2["u"] * fold_coeff2,
                           state1["vtheta_dpi"] * fold_coeff1 + state2["vtheta_dpi"] * fold_coeff2,
                           state1["dpi"] * fold_coeff1 + state2["dpi"] * fold_coeff2,
                           state1["phi_surf"],
                           state1["grad_phi_surf"],
                           state1["phi_i"] * fold_coeff1 + state2["phi_i"] * fold_coeff2,
                           state1["w_i"] * fold_coeff1 + state2["w_i"] * fold_coeff2)


#
# def accumulate_avg_explicit_terms(averaging_weight, state_c0, tracer_struct):
#   return wrap_tracer_avg_struct(tracer_struct["avg_u"] + averaging_weight *
#                                 state_c0["u"] *
#                                 state_c0["dpi"][:, :, :, :, jnp.newaxis],
#                                 tracer_struct["avg_dpi"],
#                                 tracer_struct["avg_dpi_dissip"])


@jit
def advance_state(states, coeffs):
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
  state_out = rfold_state(states[0],
                          states[1],
                          coeffs[0],
                          coeffs[1])
  for coeff_idx in range(2, len(states)):
    state_out = rfold_state(state_out,
                            states[coeff_idx],
                            1.0,
                            coeffs[coeff_idx])
  return state_out


@partial(jit, static_argnames=["dims", "hydrostatic", "deep"])
def advance_euler(state_in, dt, h_grid, v_grid, config, dims, hydrostatic=True, deep=False):
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
  u_tend = explicit_tendency(state_in, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = project_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u1 = advance_state([state_in, u_tend_c0], [1.0, dt])
  u1_cons = correct_state(u1, dt, config, hydrostatic=hydrostatic, deep=deep)
  return u1_cons


@partial(jit, static_argnames=["dims", "n_subcycle", "hydrostatic"])
def advance_euler_hypervis(state_in, dt, h_grid, v_grid, config, dims, ref_state, n_subcycle=1, hydrostatic=True):
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
  state_out = state_in
  for _ in range(n_subcycle):
    hypervis_rhs = hypervis_terms(state_in, ref_state,
                                  h_grid, dims,
                                  config,
                                  hydrostatic=hydrostatic)
    state_out = advance_state([state_in, hypervis_rhs], [1.0, dt / n_subcycle])
  return state_out


@partial(jit, static_argnames=["dims", "n_subcycle_sponge", "n_sponge", "hydrostatic"])
def advance_euler_sponge(state_in, dt, h_grid, v_grid, config, dims, n_subcycle_sponge=2, n_sponge=5, hydrostatic=True):
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
  state_out = state_in
  for _ in range(n_subcycle_sponge):
    state_out = sponge_layer(state_out,
                             .001 * dt / float(n_subcycle_sponge),
                             h_grid,
                             v_grid,
                             config,
                             dims,
                             n_sponge=n_sponge,
                             hydrostatic=hydrostatic)
  return state_out


@partial(jit, static_argnames=["dims", "hydrostatic", "deep"])
def ullrich_5stage(state_in, dt, h_grid, v_grid, config, dims, hydrostatic=True, deep=False):
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
  u_tend = explicit_tendency(state_in, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = project_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u1 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0])
  u1 = correct_state(u1, dt / 5.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u1, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = project_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u2 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0])
  u2 = correct_state(u2, dt / 5.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u2, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = project_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u3 = advance_state([state_in, u_tend_c0], [1.0, dt / 3.0])
  u3 = correct_state(u3, dt / 3.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u3, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = project_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u4 = advance_state([state_in, u_tend_c0], [1.0, 2.0 * dt / 3.0])
  u4 = correct_state(u4, 2.0 * dt / 3.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u4, h_grid, v_grid, config, hydrostatic=hydrostatic)
  u_tend_c0 = project_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  final_state = advance_state([state_in, u1, u_tend_c0], [-1.0 / 4.0,
                                                          5.0 / 4.0,
                                                          3.0 * dt / 4.0])
  final_state = correct_state(final_state, 2.0 * dt / 3.0, config, hydrostatic=hydrostatic, deep=deep)

  return final_state


def get_cfl(h_grid, v_grid, physics_config, diffusion_config, dims, sphere=True):
  #
  # estimate various CFL limits
  # Credit: This is basically copy-pasted from CAM-SE/HOMME

  # Courtesy of Paul Ullrich, Jared Whitehead
  lambda_max = {3: 1.5,
                4: 2.74,
                5: 4.18,
                6: 5.86,
                7: 7.79,
                8: 10.0}

  lambda_vis = {3: 12.0,
                4: 30.0,
                5: 91.6742,
                6: 190.117,
                7: 374.7788,
                8: 652.3015}

  npt = dims["npt"]
  scale_inv = 1.0/physics_config["radius_earth"] if sphere else 1.0

  assert npt in lambda_max.keys() and npt in lambda_vis.keys(), "Stability characteristics not calculated for {npt}"

  minimum_gauss_weight = jnp.min(h_grid["gll_weights"])

  hypervis_scaling = h_grid["hypervis_scaling"]

  max_norm_jac_inv, max_min_dx, min_min_dx = get_global_grid_defomation_metrics(h_grid, dims)

  # tensorHV.  New eigenvalues are the eigenvalues of the tensor V
  # formulas here must match what is in cube_mod.F90
  # for tensorHV, we scale out the rearth dependency
  lam = max_norm_jac_inv**2

  norm_jac_inv_hvis_tensor = (lambda_vis**2) * (max_norm_jac_inv**4) * (lam**(-hypervis_scaling/2.0))

  norm_jac_inv_hvis_const = (lambda_vis**2) * (1.0 / physics_config["radius_earth"] * max_norm_jac_inv)**4
  norm_jac_inv_hvis = norm_jac_inv_hvis_tensor if "tensor_hypervis" in diffusion_config.keys() else norm_jac_inv_hvis_const
  rkssp_euler_stability = minimum_gauss_weight / (120.0 * max_norm_jac_inv * scale_inv)
  rk2_tracer = 1.0 / (120.0 * max_norm_jac_inv * lambda_max * scale_inv)
  gravit_wave_stability = 1.0 / (342.0 * max_norm_jac_inv * lambda_max * scale_inv)
  hypervis_stability_dpi = 1.0 / (diffusion_config["nu_dpi"] * norm_jac_inv_hvis)
  hypervis_stability_vort = 1.0 / (diffusion_config["nu"] * norm_jac_inv_hvis)
  hypervis_stability_div = 1.0 / (diffusion_config["nu_div"] * norm_jac_inv_hvis)
  nu_top_max = jnp.max(diffusion_config["nu_ramp"]) * diffusion_config["nu_top"]
  sponge_layer_stab = 1.0 / (nu_top_max*((scale_inv*max_norm_jac_inv)**2)*lambda_vis)
  return {"dt_rkssp_euler": rkssp_euler_stability,
          "dt_rk2_tracer": rk2_tracer,
          "dt_gravity_wave": gravit_wave_stability,
          "dt_hypervis_scalar": hypervis_stability_dpi,
          "dt_hypervis_vort": hypervis_stability_vort,
          "dt_hypervis_div": hypervis_stability_div,
          "dt_sponge_layer": sponge_layer_stab}


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
                        sponge_steps_per_dyn=-1):
  cfl_info = get_cfl(h_grid, v_grid, physics_config, diffusion_config)
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
                    



