from ..config import jnp

def clip_and_sum_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass):
  # c -> scaled_mass
  # x -> tracer
  scaled_mass = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  tracer = tracer_mass_tend / d_mass
  sum_scaled_mass = jnp.max(scaled_mass, axis=(1, 2))
  sum_scaled_tracer = jnp.max(tracer * scaled_mass, axis=(1, 2))
  tracer_min = jnp.where(sum_scaled_tracer < tracer_min * sum_scaled_mass, sum_scaled_tracer / sum_scaled_mass, tracer_min)
  tracer_max = jnp.where(sum_scaled_tracer > tracer_max * sum_scaled_mass, sum_scaled_tracer / sum_scaled_mass, tracer_max)
  add_mass = jnp.zeros_like(tracer_mass_tend)
  mask_overshoot = tracer > tracer_max[:, jnp.newaxis, jnp.newaxis, :]
  add_mass = jnp.where(mask_overshoot, 
                       add_mass + (tracer - tracer_max[:, jnp.newaxis, jnp.newaxis, :]) * scaled_mass,
                       add_mass)
  tracer = jnp.where(mask_overshoot,
                     tracer_max[:, jnp.newaxis, jnp.newaxis, :],
                     tracer)
  mask_undershoot = tracer < tracer_min[:, jnp.newaxis, jnp.newaxis, :]
  add_mass = jnp.where(mask_undershoot,
                       add_mass + (tracer - tracer_min[:, jnp.newaxis, jnp.newaxis, :]) * scaled_mass,
                       add_mass)
  tracer = jnp.where(mask_undershoot,
                     tracer_min[:, jnp.newaxis, jnp.newaxis, :],
                     tracer)
  add_mass_per_lev = jnp.sum(add_mass, axis=(1, 2))
  modified = jnp.abs(add_mass_per_lev) > 0.0
  tracer_adjustment = jnp.where(add_mass > 0.0,
                                tracer_max[:, jnp.newaxis, jnp.newaxis, :] - tracer,
                                jnp.zeros_like(tracer))
  tracer_adjustment = jnp.where(add_mass < 0.0,
                                tracer - tracer_min[:, jnp.newaxis, jnp.newaxis, :],
                                tracer_adjustment)
  denominator = jnp.sum(tracer_adjustment * scaled_mass, axis=(1, 2))
  do_mass_adjustment = jnp.logical_and(modified, denominator > 0.0)
  print(add_mass_per_lev[:, jnp.newaxis, jnp.newaxis, :].shape)
  tracer = jnp.where(do_mass_adjustment[:, jnp.newaxis, jnp.newaxis, :],
                     tracer + (add_mass_per_lev/denominator)[:, jnp.newaxis, jnp.newaxis, :] * tracer_adjustment,
                     tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out