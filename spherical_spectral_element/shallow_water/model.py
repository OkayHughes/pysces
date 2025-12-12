from ..config import jnp, jit, versatile_assert, wrapper
from ..assembly import dss_scalar
from ..operators import sphere_vorticity, sphere_gradient, sphere_divergence
from ..operators import sphere_laplacian_wk, sphere_vec_laplacian_wk
from functools import partial


def create_state_struct(u, h, hs):
  return {"u": u,
          "h": h,
          "hs": hs}


def get_config_sw(radius_earth=6371e3, earth_period=7.292e-5, gravity=9.81, alpha=0.0, ne=30):
  return {"radius_earth": wrapper(radius_earth),
          "earth_period": wrapper(earth_period),
          "gravity": wrapper(gravity),
          "alpha": wrapper(alpha),
          "nu": wrapper(2.5e15 * ((30.0 * radius_earth) / (ne * 6371e3))**3.2)}


@partial(jit, static_argnames=["dims"])
def dss_state(state, grid, dims):
  u = dss_scalar(state["u"][:, :, :, 0], grid, dims)
  v = dss_scalar(state["u"][:, :, :, 1], grid, dims)
  h = dss_scalar(state["h"][:, :, :], grid, dims)
  return create_state_struct(jnp.stack((u, v), axis=-1), h, state["hs"])


@jit
def calc_rhs(state_in, grid, config):
  coriolis = (-jnp.cos(grid["physical_coords"][:, :, :, 1]) *
              jnp.cos(grid["physical_coords"][:, :, :, 0]) *
              jnp.sin(config["alpha"]) +
              jnp.sin(grid["physical_coords"][:, :, :, 0]) *
              jnp.cos(config["alpha"]))
  abs_vort = sphere_vorticity(state_in["u"], grid, a=config["radius_earth"]) + 2 * config["earth_period"] * coriolis
  energy = 0.5 * (state_in["u"][:, :, :, 0]**2 +
                  state_in["u"][:, :, :, 1]**2) + config["gravity"] * (state_in["h"] + state_in["hs"])
  energy_grad = sphere_gradient(energy, grid, a=config["radius_earth"])
  u_tend = abs_vort * state_in["u"][:, :, :, 1] - energy_grad[:, :, :, 0]
  v_tend = -abs_vort * state_in["u"][:, :, :, 0] - energy_grad[:, :, :, 1]
  h_tend = -sphere_divergence(state_in["h"][:, :, :, jnp.newaxis] * state_in["u"], grid, a=config["radius_earth"])
  return create_state_struct(jnp.stack((u_tend, v_tend), axis=-1), h_tend, state_in["hs"])


@partial(jit, static_argnames=["dims"])
def calc_hypervis(state_in, grid, config, dims):
  a = config["radius_earth"]
  u_tmp = sphere_vec_laplacian_wk(state_in["u"], grid, a=a, damp=True)
  h_tmp = sphere_laplacian_wk(state_in["h"][:, :, :], grid, a=a)
  lap1 = dss_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims)
  u_tmp = -config["nu"] * sphere_vec_laplacian_wk(lap1["u"], grid, a=a, damp=True)
  h_tmp = -config["nu"] * sphere_laplacian_wk(lap1["h"], grid, a=a)
  return dss_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims)


@partial(jit, static_argnames=["dims", "substeps"])
def advance_hypervis_euler(state_in, dt, grid, config, dims, substeps=1):
  next_step = state_in
  for k in range(substeps):
    hvis_tend = calc_hypervis(next_step, grid, config, dims)
    next_step = advance_state([next_step, hvis_tend], [1.0, dt / substeps])
  return next_step


@jit
def advance_state(states_in, coeffs):
  state_res = create_state_struct(states_in[0]["u"] * coeffs[0],
                                  states_in[0]["h"] * coeffs[0],
                                  states_in[0]["hs"])
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = create_state_struct(state_res["u"] + state["u"] * coeff,
                                    state_res["h"] + state["h"] * coeff,
                                    state_res["hs"])
  return state_res


@partial(jit, static_argnames=["dims"])
def advance_step_euler(state_in, dt, grid, config, dims):
  state_tend = calc_rhs(state_in, grid, config)
  state_tend_c0 = dss_state(state_tend, grid, dims)
  return advance_state([state_in, state_tend_c0], [1.0, dt])


@partial(jit, static_argnames=["dims"])
def advance_step_ssprk3(state0, dt, grid, config, dims):
  tend = calc_rhs(state0, grid, config)
  tend_c0 = dss_state(tend, grid, dims)
  state1 = advance_state([state0, tend_c0], [1.0, dt])
  tend = calc_rhs(state1, grid, config)
  tend_c0 = dss_state(tend, grid, dims)
  state2 = advance_state([state0, state1, tend_c0], [3.0 / 4.0,
                                                     1.0 / 4.0,
                                                     1.0 / 4.0 * dt])
  tend = calc_rhs(state2, grid, config)
  tend_c0 = dss_state(tend, grid, dims)
  return advance_state([state0, state2, tend_c0], [1.0 / 3.0,
                                                   2.0 / 3.0,
                                                   2.0 / 3.0 * dt])


# @partial(jit, static_argnames=["dims", "diffusion", "ne", "end_time", "step_type"])
def simulate_sw(end_time, ne, state_in, grid, config, dims, diffusion=False, step_type="ssprk3"):
  dt = 100.0 * (30.0 / ne)  # todo: automatically calculate CFL from sw dispersion relation
  state_n = state_in
  t = 0.0
  times = jnp.arange(0.0, end_time, dt)
  k = 0
  for t in times:
    print(f"{k/len(times-1)*100}%")
    if step_type == "ssprk3":
      state_tmp = advance_step_ssprk3(state_n, dt, grid, config, dims)
    elif step_type == "euler":
      state_tmp = advance_step_euler(state_n, dt, grid, config)

    if diffusion:
      state_np1 = advance_hypervis_euler(state_tmp, dt, grid, config, dims, substeps=1)
    else:
      state_np1 = state_tmp
    state_n, state_np1 = state_np1, state_n

    versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["u"]))))
    versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  return state_n
