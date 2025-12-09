from ..config import jnp

def zerroukat_remap(Qdp, dpi_model, dpi_reference, num_lev, filter=False, tiny=1e-12, qmax=1e50):
  # assumes
  pi_int_reference = jnp.concatenate((jnp.zeros_like(dpi_reference[:, :, :, 0:1]), jnp.cumsum(dpi_reference, axis=-1)), axis=-1)
  pi_int_model = jnp.concatenate((jnp.zeros_like(dpi_model[:, :, :, 0:1]), jnp.cumsum(dpi_model, axis=-1)), axis=-1)
  values_model = jnp.concatenate((jnp.zeros_like(Qdp[:, :, :, 0:1, :]),
                                 jnp.cumsum(Qdp, axis=-2)), axis=-2)

  # binary search
  # idxs is model to reference
  idxs = jnp.zeros_like(pi_int_reference[:, :, :, 1:-1], dtype=jnp.float32)
  frac = 0.5
  axis_size = 1.0 * num_lev
  for _ in range(8):
    levels_model = jnp.take_along_axis(pi_int_model, jnp.floor(idxs).astype(jnp.int32), axis=-1)
    levels_model_below = jnp.take_along_axis(pi_int_model, jnp.floor(idxs).astype(jnp.int32)+1, axis=-1)
    low_enough = pi_int_reference[:, :, :, 1:-1] > levels_model
    too_low = pi_int_reference[:, :, :, 1:-1] > levels_model_below
    converged = jnp.logical_and(low_enough,
                                jnp.logical_not(too_low))
    jump = frac * axis_size
    idxs = jnp.where(jnp.logical_not(converged),
                     jnp.where(too_low, idxs + jump, idxs - jump),
                     idxs)
    frac *= 0.5
  assert(jnp.all(converged))
  idxs = jnp.floor(idxs).astype(jnp.int32)
  idxs = jnp.concatenate((jnp.zeros_like(idxs[:, :, :, 0:1]),
                          idxs,
                          (num_lev-1) * jnp.ones_like(idxs[:, :, :, 0:1])), axis=-1)
  model_above = jnp.take_along_axis(pi_int_model, idxs, axis=-1)
  model_below = jnp.take_along_axis(pi_int_model, idxs+1, axis=-1)

  zgam = (pi_int_reference - model_above) / (model_below - model_above)
  zgam[:, :, :, 0] = 0.0
  zgam[:, :, :, -1] = 1.0

  zhdp = pi_int_model[:, :, :, 1:] - pi_int_model[:, :, :, :-1]

  h = 1/zhdp

  zarg = Qdp * h[:, :, :, :, jnp.newaxis]
  brc = jnp.ones((1, 1, 1, 1, Qdp.shape[4]))
  diag_top = 2.0 * jnp.ones_like(zarg[:, :, :, 0:1, :]) * brc
  diag_mid = 2.0 * (h[:, :, :, 1:, jnp.newaxis] + h[:, :, :, :-1, jnp.newaxis]) * brc
  diag_bottom = 2.0 * jnp.ones_like(zarg[:, :, :, 0:1, :]) * brc

  rhs_top = 3.0 * zarg[:, :, :, 0:1, :]
  rhs_mid = 3.0 * (zarg[:, :, :, 1:, :] * h[:, :, :, 1:, jnp.newaxis] +
               zarg[:, :, :, :-1, :] * h[:, :, :, :-1, jnp.newaxis])
  rhs_bottom = 3.0 * zarg[:, :, :, -1:, :]
  rhs_base = jnp.concatenate((rhs_top/diag_top, rhs_mid, rhs_bottom), axis=-2)

  lower_diag_top = jnp.ones_like(zarg[:, :, :, 0:1, :])
  lower_diag_mid = h[:, :, :, :-1, jnp.newaxis] * brc
  lower_diag_bottom = jnp.ones_like(zarg[:, :, :, -1:, :])
  lower_diag = jnp.concatenate((lower_diag_top, lower_diag_mid, lower_diag_bottom), axis=-2)
  upper_diag_top = jnp.ones_like(zarg[:, :, :, 0:1, :])
  upper_diag_mid = h[:, :, :, 1:, jnp.newaxis] * brc
  upper_diag_bottom = jnp.zeros_like(zarg[:, :, :, -1:, :])

  upper_diag = jnp.concatenate((upper_diag_top, upper_diag_mid, upper_diag_bottom), axis=-2)

  diag = jnp.concatenate((diag_top, diag_mid, diag_bottom), axis=-2)
  q_diag = [-upper_diag_top[:, :, :, 0, :] / diag_top[:, :, :, 0, :]]
  rhs = [rhs_base[:, :, :, 0, :]]
  # these are necessarily a fold
  for k_idx in range(1, num_lev+1):
    denom = 1.0 / (diag[:, :, :, k_idx, :] + lower_diag[:, :, :, k_idx, :] * q_diag[-1])
    q_diag.append(-upper_diag[:, :, :, k_idx, :] * denom)
    rhs.append((rhs_base[:, :, :, k_idx, :] - lower_diag[:, :, :, k_idx, :] * rhs[-1]) * denom)
  rhs_final = [rhs[-1]]
  # these are necessarily a fold
  for k_idx in reversed(range(0, num_lev)):
    rhs_final.append(rhs[k_idx] + q_diag[k_idx] * rhs_final[-1])
  rhs = jnp.stack([x for x in reversed(rhs_final)], axis=-2)
  
  if filter:
    filter_code = []
    dy = jnp.concatenate((zarg[:, :, :, 1:, :] - zarg[:, :, :, :-1, :],
                          zarg[:, :, :, -1:, :] - zarg[:, :, :, -2:-1, :]), axis=-2)
    dy = jnp.where(jnp.abs(dy) < tiny, 0.0, dy)
    lev = lambda arr, j: arr[:, :, :, j, :]
    ones = jnp.ones_like(zarg[:, :, :, 0, :], dtype=jnp.int32)
    ones_f = jnp.ones_like(zarg[:, :, :, 0, :])

    zeros = jnp.zeros_like(zarg[:, :, :, 0, :], dtype=jnp.int32)
    for k in range(num_lev):
      im1 = max(0, k-1)
      im2 = max(0, k-2)
      im3 = max(0, k-3)
      ip1 = min(num_lev-1, k+1)
      t1 = jnp.where((lev(zarg, k) - lev(rhs, k)) *
                     (lev(rhs, k) - lev(zarg, im1)) >= 0, ones, zeros)
      cond1 = lev(dy, im2) * (lev(rhs, k) - lev(zarg, im1)) > 0
      cond2 = lev(dy, im2) * lev(dy, im3) > 0
      cond3 = lev(dy, k) * lev(dy, ip1) > 0
      cond4 = lev(dy, im2) * lev(dy, k) < 0
      t2 = jnp.where(cond1 * cond2 * cond3 * cond4 == 1, ones, zeros)
      t3 = jnp.where(lev(rhs, k) - lev(zarg, im1) > jnp.abs(lev(rhs, k) - lev(zarg, k)), ones, zeros)
      filter_code.append(jnp.where(t1 + t2 > 0, zeros, ones))
      rhs[:, :, :, k, :] = ((1.0 - filter_code[k]) * lev(rhs, k) +
                            filter_code[k] * (t3 * lev(zarg, k) + (1.0-t3) * lev(zarg, im1)))
      filter_code[im1] = jnp.maximum(filter_code[im1], filter_code[k])
    rhs = jnp.where(rhs > qmax, qmax, rhs)
    rhs = jnp.where(rhs < 0, 0.0, rhs)
    za0_base = rhs[:, :, :, :-1, :]
    za1_base = -4.0 * rhs[:, :, :, :-1, :] - 2.0 * rhs[:, :, :, 1:, :] + 6 * zarg
    za2_base = 3.0 * rhs[:, :, :, :-1, :] + 3.0 * rhs[:, :, :, 1:, :] - 6 * zarg

    za0 = [rhs[:, :, :, k, :] for k in range(num_lev)]
    za1 = [-4.0 * rhs[:, :, :, k, :] - 2.0 * rhs[:, :, :, k+1, :] + 6 * zarg[:, :, :, k, :] for k in range(num_lev)]
    za2 = [3.0 * rhs[:, :, :, k, :] + 3.0 * rhs[:, :, :, k+1, :] - 6 * zarg[:, :, :, k, :] for k in range(num_lev)]
    dy = rhs[:, :, :, 1:, :] - rhs[:, :, :, :-1, :]
    dy = jnp.where(jnp.abs(dy) < tiny, 0.0, dy)
    
    h = rhs[:, :, :, 1:, :]

    for k in range(num_lev):
      xm_d = jnp.where(jnp.abs(za2[k]) < tiny, 1.0 * ones_f, 2 * za2[k] )
      xm = jnp.where(jnp.abs(za2[k]) < tiny, 0.0 * ones_f, -za1[k]/xm_d)
      f_xm = za0[k] + za1[k] * xm + za2[k] * xm**2
      t1 = jnp.where(jnp.abs(za2[k]) > tiny, ones, zeros)
      t2 = jnp.where(jnp.logical_or((xm <= 0), (xm >= 1)), ones, zeros )
      t3 = jnp.where(za2[k] > 0, ones, zeros)
      t4 = jnp.where(za2[k] < 0, ones, zeros)
      tm = jnp.where(t1 * ((1-t2) + t3) == 2, ones, zeros)
      tp = jnp.where(t1 * ((1 - t2) + (1-t3) + t4) == 3, ones, zeros)
      peaks = jnp.where(tm == 1, -1 * ones, zeros)
      peaks = jnp.where(tp == 1, ones, peaks)
      peaks_min = jnp.where(tm == 1, f_xm, jnp.minimum(za0[k], za0[k] + za1[k] + za2[k]))
      peaks_max = jnp.where(tp == 1, f_xm, jnp.maximum(za0[k], za0[k] + za1[k] + za2[k]))
      im1 = max(0, k-1)
      im2 = max(0, k-2)
      ip1 = min(num_lev-1, k+1)
      ip2 = min(num_lev-1, k+2)
      cond1 = lev(dy, im2) * lev(dy, im1) <= tiny
      cond2 = lev(dy, ip1) * lev(dy, ip2) <= tiny
      cond3 = lev(dy, im1) * lev(dy, ip1) >= tiny
      cond4 = lev(dy, im1) * peaks <= tiny
      t1 = jnp.where(cond1 + cond2 + cond3 + cond4 > 0, jnp.abs(peaks), zeros)
      cond1 = lev(rhs, k) >= qmax
      cond2 = lev(rhs, k) <= 0
      cond3 = peaks_max > qmax
      cond4 = peaks_min < tiny
      filter_code[k] = jnp.where(cond1 + cond2 + cond3 + cond4, ones, t1 + (1-t1) * filter_code[k])

      level1 = lev(rhs, k)
      level2 = (2.0 * lev(rhs, k) + lev(h, k))/3.0
      level3 = 0.5 * (lev(rhs, k) + lev(h, k))
      level4 = 1.0/3.0 * lev(rhs, k) + 2.0 * (1.0/3.0) * lev(h, k)
      level5 = lev(h, k)
      
      t1 = jnp.where(lev(h, k) >= lev(rhs, k), ones, zeros)
      t2 = jnp.where(jnp.logical_or(lev(zarg, k) <= level1,
                                    lev(zarg, k) >= level5), ones, zeros)
      t3 = jnp.where(jnp.logical_and(lev(zarg, k) > level1,
                                     lev(zarg, k) < level2), ones, zeros)
      t4 = jnp.where(jnp.logical_and(lev(zarg, k) > level4,
                                     lev(zarg, k) < level5), ones, zeros)
      lt1 = t1 * t2
      lt2 = t1 * (1 - t2 + t3)
      lt3 = t1 * (1 - t2 + 1 - t3 + t4)
      za0[k] = jnp.where(lt1 == 1, lev(zarg, k), za0[k])
      za1[k] = jnp.where(lt1 == 1, 0.0 * ones_f, za1[k])
      za2[k] = jnp.where(lt1 == 1, 0.0 * ones_f, za2[k])

      za0[k] = jnp.where(lt2 == 2, lev(rhs, k), za0[k])
      za1[k] = jnp.where(lt2 == 2, 0.0 * ones_f, za1[k])
      za2[k] = jnp.where(lt2 == 2, 3 * (lev(zarg, k) - lev(rhs, k)), za2[k])

      za0[k] = jnp.where(lt3 == 3, -2.0 * lev(h, k) + 3.0 * lev(zarg, k), za0[k])
      za1[k] = jnp.where(lt3 == 3, 6.0 * lev(h, k) - 6.0 * lev(zarg, k), za1[k])
      za2[k] = jnp.where(lt3 == 3, -3.0 * lev(h, k) + 3.0 * lev(zarg, k), za2[k])

      t2 = jnp.where(jnp.logical_or(lev(zarg, k) >= level1,
                                    lev(zarg, k) <= level5), ones, zeros)
      t3 = jnp.where(jnp.logical_and(lev(zarg, k) < level1,
                                     lev(zarg, k) > level2), ones, zeros)
      t4 = jnp.where(jnp.logical_and(lev(zarg, k) < level4,
                                     lev(zarg, k) > level5), ones, zeros)
      lt1 = (1 - t1) * t2
      lt2 = (1 - t1) * (1 - t2 + t3)
      lt3 = (1 - t1) * (1 - t2 + 1 - t3 + t4)

      za0[k] = jnp.where(lt1 == 1, lev(zarg, k), za0[k])
      za1[k] = jnp.where(lt1 == 1, 0.0 * ones_f, za1[k])
      za2[k] = jnp.where(lt1 == 1, 0.0 * ones_f, za2[k])

      za0[k] = jnp.where(lt2 == 2, lev(rhs, k), za0[k])
      za1[k] = jnp.where(lt2 == 2, 0.0 * ones_f, za1[k])
      za2[k] = jnp.where(lt2 == 2, 3.0 * (lev(zarg, k) - lev(rhs, k)), za2[k])

      za0[k] = jnp.where(lt3 == 3, -2.0 * lev(h, k) + 3 * lev(zarg, k), za0[k])
      za1[k] = jnp.where(lt3 == 3, 6.0 * lev(h, k) - 6.0 * lev(zarg, k), za1[k])
      za2[k] = jnp.where(lt3 == 3, -3.0 * lev(h, k) + 3.0 * lev(zarg, k), za2[k])

    za0 = jnp.where(jnp.stack(filter_code, axis=-2) > 0, 
                    jnp.stack(za0, axis=-2),
                    za0_base)
    za1 = jnp.where(jnp.stack(filter_code, axis=-2) > 0, 
                    jnp.stack(za1, axis=-2),
                    za1_base)
    za2 = jnp.where(jnp.stack(filter_code, axis=-2) > 0, 
                    jnp.stack(za2, axis=-2),
                    za2_base)
  else:
    za0 = rhs[:, :, :, :-1, :]
    za1 = -4.0 * rhs[:, :, :, :-1, :] - 2.0 * rhs[:, :, :, 1:, :] + 6 * zarg
    za2 = 3.0 * rhs[:, :, :, :-1, :] + 3.0 * rhs[:, :, :, 1:, :] - 6 * zarg

  zhdp_mapped = jnp.take_along_axis(zhdp, idxs[:, :, :, 1:], axis=-1)[:, :, :, :, jnp.newaxis]
  zv1 = jnp.zeros_like(Qdp[:, :, :, 0, :])
  zv_mapped = jnp.take_along_axis(values_model[:, :, :, :-1, :], idxs[:, :, :, 1:, jnp.newaxis], axis=-2)
  za0_mapped = jnp.take_along_axis(za0[:, :, :, :, :], idxs[:, :, :, 1:, jnp.newaxis], axis=-2)
  za1_mapped = jnp.take_along_axis(za1[:, :, :, :, :], idxs[:, :, :, 1:, jnp.newaxis], axis=-2)
  za2_mapped = jnp.take_along_axis(za2[:, :, :, :, :], idxs[:, :, :, 1:, jnp.newaxis], axis=-2)

  Qdp_out = []
  for k_idx in range(num_lev):
    zv2 = zv_mapped[:, :, :, k_idx, :] + (za0_mapped[:, :, :, k_idx, :] *
                                          zgam[:, :, :, k_idx+1, jnp.newaxis] +
                                          za1_mapped[:, :, :, k_idx, :]/2.0 * zgam[:, :, :, k_idx+1, jnp.newaxis]**2 +
                                          za2_mapped[:, :, :, k_idx, :]/3.0 * zgam[:, :, :, k_idx+1, jnp.newaxis]**3 )* zhdp_mapped[:, :, :, k_idx, :]
    Qdp_out.append(zv2 - zv1)
    zv1 = zv2
  return jnp.stack(Qdp_out, axis=-2)
