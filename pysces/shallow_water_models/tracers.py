from ..config import jit, jnp
from ..operations_2d.operators import horizontal_divergence
from ..tracer_transport.eulerian_spectral import advance_tracers_rk2
from functools import partial

@jit
def stack_tracers_shallow_water(tracer_like):
  tracer_names = {}
  tracer_mass_flat = []
  ct = 0
  for tracer_name in tracer_like.keys():
    tracer_names[tracer_name] = ct
    tracer_mass_flat.append(tracer_like[tracer_name])
    ct += 1
  return jnp.stack(tracer_mass_flat, axis=0)[:, :, :, :, jnp.newaxis], tracer_names


@jit
def unstack_tracers_shallow_water(tracer_like, tracer_names):
  tracers = {}
  for tracer_name, tracer_idx in tracer_names.items():
    tracers[tracer_name] = tracer_like[tracer_idx, :, :, :, 0]
  return tracers


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_tracers_shallow_water(tracers,
                                  tracer_consist_dyn,
                                  tracer_init_struct,
                                  grid,
                                  dims,
                                  physics_config,
                                  diffusion_config,
                                  timestep_config,
                                  tracer_consist_hypervis=None):
  d_mass_init = tracer_init_struct["d_mass_init"]
  d_mass_end = tracer_init_struct["d_mass_end"]
  u_d_mass_avg = tracer_consist_dyn["u_d_mass_avg"]
  d_mass_dyn_tend = -horizontal_divergence(u_d_mass_avg, grid, a=physics_config["radius_earth"])
  if tracer_consist_hypervis is not None:
    d_mass_hypervis_tend = tracer_consist_hypervis["d_mass_hypervis_tend"][:, :, :, jnp.newaxis]
    d_mass_hypervis_avg = tracer_consist_hypervis["d_mass_hypervis_avg"][:, :, :, jnp.newaxis]
  else:
    d_mass_hypervis_tend = None
    d_mass_hypervis_avg = None
  stacked_tracers, tracer_names = stack_tracers_shallow_water(tracers)
  stacked_tracer_mass = stacked_tracers * d_mass_init[jnp.newaxis, :, :, :, jnp.newaxis]
  stacked_tracer_mass_out = advance_tracers_rk2(stacked_tracer_mass,
                                                d_mass_init[:, :, :, jnp.newaxis],
                                                u_d_mass_avg[:, :, :, jnp.newaxis, :],
                                                d_mass_dyn_tend[:, :, :, jnp.newaxis],
                                                grid,
                                                physics_config,
                                                diffusion_config,
                                                timestep_config,
                                                dims,
                                                d_mass_hypervis_tend=d_mass_hypervis_tend,
                                                d_mass_hypervis_avg=d_mass_hypervis_avg)
  stacked_tracer_out = stacked_tracer_mass_out / d_mass_end[jnp.newaxis, :, :, :, jnp.newaxis]
  return unstack_tracers_shallow_water(stacked_tracer_out, tracer_names)