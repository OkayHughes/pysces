from ...config import jit
from functools import partial
from ..operators_3d import horizontal_divergence_3d
from ..model_state import project_scalar_3d, advance_tracers, wrap_tracer_mass, wrap_tracers
from ...model_info import cam_se_models, homme_models


@partial(jit, static_argnames=["model"])
def flatten_tracer_like(tracers, model):
  tracers_flat = {}
  for species_name in tracers["moisture_species"].keys():
    tracers_flat[species_name] = tracers["moisture_species"][species_name]
  for species_name in tracers["tracers"].keys():
    tracers_flat[species_name] = tracers["tracers"][species_name]
  if model in cam_se_models:
    for species_name in tracers["dry_air_species"].keys():
      tracers_flat[species_name] = tracers["dry_air_species"][species_name]


@partial(jit, static_argnames=["model"])
def tracer_to_mass(d_mass, tracers, model):
  moisture_mass = {}
  for species_name in tracers["moisture_species"].keys():
      moisture_mass[species_name] = d_mass * tracers["moisture_species"][species_name]
  tracers_mass = {}
  for species_name in tracers["tracers"].keys():
    tracers_mass[species_name] = d_mass * tracers["tracers"][species_name]
  if model in cam_se_models:
    dry_air_species_mass = {}
    for species_name in tracers["dry_air_species"].keys():
        dry_air_species_mass[species_name] = d_mass * tracers["dry_air_species"][species_name]
  else:
    dry_air_species_mass = None
  return wrap_tracer_mass(moisture_mass,
                          tracers_mass,
                          model,
                          dry_air_species_mass=dry_air_species_mass)


@partial(jit, static_argnames=["model"])
def mass_to_tracer(d_mass, tracers_mass, model):
  moisture_mixing_ratio = {}
  for species_name in tracers_mass["moisture_species"].keys():
      moisture_mixing_ratio[species_name] = tracers_mass["moisture_species"][species_name] / d_mass
  tracers_mixing_ratio = {}
  for species_name in tracers_mass["tracers"].keys():
    tracers_mixing_ratio[species_name] = tracers_mass["tracers"][species_name] / d_mass
  if model in cam_se_models:
    dry_air_species_mixing_ratio = {}
    for species_name in tracers_mass["dry_air_species"].keys():
        dry_air_species_mixing_ratio[species_name] = tracers_mass["dry_air_species"][species_name] / d_mass
  else:
    dry_air_species_mixing_ratio = None
  return wrap_tracer_mass(moisture_mixing_ratio,
                          tracers_mixing_ratio,
                          model,
                          dry_air_species_mass=dry_air_species_mixing_ratio)


@partial(jit, static_argnames=["timestep_config", "model", "dims"])
def advect_dissipate_limit_tracers_rk2(tracer_state, tracer_consist_dyn, tracer_consist_begin, physics_config, diffusion_config, timestep_config, h_grid, model, dims):
  divergence_d_mass_term = -horizontal_divergence_3d(tracer_consist_dyn["u_d_mass_tendency"], h_grid, physics_config)
  divergence_d_mass_term = project_scalar_3d(divergence_d_mass_term, h_grid, dims)
  rk_stages = 3.0
  if "diffuse_d_mass" in diffusion_config.keys():
    tracer_like = tracer_to_mass(idk_yet, tracer_state, model)
  else:
    tracer_like = tracer_state
  tracer_like_tmp = euler_step(tracer_like,
                                  timestep_config["tracer_advection"]["dt"] / 2.0,
                                  divergence_d_mass_term,
                                  tracer_consist_visc, 
                                  tracer_consist_begin,
                                  0.0)
  tracer_like_tmp = euler_step(tracer_like_tmp,
                                timestep_config["tracer_advection"]["dt"] / 2.0,
                                divergence_d_mass_term,
                                tracer_consist_visc, 
                                tracer_consist_begin,
                                1.0)
  tracer_like_last = euler_step(tracer_like_tmp,
                                timestep_config["tracer_advection"]["dt"] / 2.0,
                                divergence_d_mass_term,
                                tracer_consist_visc, 
                                tracer_consist_begin,
                                2.0)
  tracers_like_after_advection = advance_tracers([tracer_like, tracer_like_last],
                                            [1.0 / rk_stages, (rk_stages-1.0)/rk_stages],
                                            model)
  
  if "diffuse_d_mass" in diffusion_config.keys():
    tracer_like_out = mass_to_tracer(idk_yet, tracers_like_after_advection, model)
  else:
    tracer_like_out = tracers_like_after_advection
  return tracer_like_out
  

@partial(jit, static_argnames=["model", "dims", "timestep_config"])
def euler_step(tracer_like_tmp, dt, tracer_consist_dyn, tracer_consist_visc, tracer_consist_begin, rhs_scale):
  # defaultly: set rhs_visc_scale = 0.0
  # calculate updated dp
  # Calculate maximum Q by dividing out by updated
  #   on rhs_mult == 0 or 2, set qmin, qmax to raw maximum of input Q
  #   on rhs_mult == 1, update qmin, qmax with any new local maxima.
  # on rhs_muly == 0
  #   * update qmin, qmax with non-local maxima
  # on rhs_mult == 2
  #   * set rhs_visc_scale = 3.0
  #   * if we're dissipating d_mass
  #     * divide out by dp0 (calculated per vert coord)  
  #   * Update qmin, qmax
  #   * calculate biharmonic (continuous)
  #    * scale by rhs_viss * dt * dp0
  # start actual advection lmao
  #   * Overwrite dp, because divdp_proj may have been projected
  #   * calculate weird vstar wind quantity, vdp_avg / intermediate_dp
  #   * calculate discont intermediate dp_what = dp + dt * divdp_discont
  #   * if d_mass is dissipated and rhs_visc_scale is non-zero
  #     * update dp_what += rhs_visc_scale * dt * nu_q * dpdiss_biharmonic
  #   * add 0 as a global minimum of qmin if we;re using the fancy limiter
  #   * calculate vstar * qdp
  #   * calculate dp_star = horiz_div(vstar * qdp) (might not actually be qdp)
  #   * add biharmonic term if rhs_visc_scale isn't 0
  #   * apply limiter
  #   * dss all tracers