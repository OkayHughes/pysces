from ..config import jnp

def clip_and_sum_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass):
  # c -> scaled_mass
  # x -> tracer
  scaled_mass = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  tracer = tracer_mass_tend / d_mass
  sum_scaled_mass = jnp.sum(scaled_mass, axis=(1, 2))
  sum_scaled_tracer = jnp.sum(tracer * scaled_mass, axis=(1, 2))
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
  add_mask = (add_mass_per_lev > 0.0)[:, jnp.newaxis, jnp.newaxis, :]
  tracer_adjustment = jnp.where(add_mask,
                                tracer_max[:, jnp.newaxis, jnp.newaxis, :] - tracer,
                                tracer - tracer_min[:, jnp.newaxis, jnp.newaxis, :])
  tracer_adjustment = jnp.where(modified[:, jnp.newaxis, jnp.newaxis, :],
                                tracer_adjustment,
                                jnp.zeros_like(tracer))
  denominator = jnp.sum(tracer_adjustment * scaled_mass, axis=(1, 2))
  do_mass_adjustment = jnp.logical_and(modified, denominator > 0.0)
  tracer = jnp.where(do_mass_adjustment[:, jnp.newaxis, jnp.newaxis, :],
                     tracer + (add_mass_per_lev/denominator)[:, jnp.newaxis, jnp.newaxis, :] * tracer_adjustment,
                     tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out

def full_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass, tol_limiter=1e-10):
  # c -> scaled_mass
  # x -> tracer
  npt = tracer_mass_tend.shape[1]
  scaled_mass = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  tracer = tracer_mass_tend / d_mass
  sum_scaled_mass = jnp.sum(scaled_mass, axis=(1, 2))
  sum_scaled_tracer = jnp.sum(tracer * scaled_mass, axis=(1, 2))
  tracer_min = jnp.where(sum_scaled_tracer < tracer_min * sum_scaled_mass, sum_scaled_tracer / sum_scaled_mass, tracer_min)
  tracer_max = jnp.where(sum_scaled_tracer > tracer_max * sum_scaled_mass, sum_scaled_tracer / sum_scaled_mass, tracer_max)
  for iter_idx in range(npt * npt-1):
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

    add_mask = (add_mass_per_lev > 0.0)[:, jnp.newaxis, jnp.newaxis, :]
    not_overshoot_mask = tracer < tracer_max[:, jnp.newaxis, jnp.newaxis, :]
    not_undershoot_mask = tracer > tracer_min[:, jnp.newaxis, jnp.newaxis, :]
    weight_sum = jnp.sum(jnp.where(jnp.logical_and(add_mask, not_overshoot_mask),
                                   scaled_mass,
                                   jnp.zeros_like(tracer)),
                         axis=(1, 2))
    

    tracer = jnp.where(jnp.logical_and(add_mask, not_overshoot_mask),
                       tracer + (add_mass_per_lev/weight_sum)[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
    not_add_mask = jnp.logical_not(add_mask)
    weight_sum = jnp.sum(jnp.where(jnp.logical_and(not_add_mask, not_undershoot_mask),
                                   scaled_mass,
                                   jnp.zeros_like(tracer)),
                         axis=(1, 2))
    tracer = jnp.where(jnp.logical_and(not_add_mask, not_undershoot_mask),
                       tracer + (add_mass_per_lev/weight_sum)[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out  