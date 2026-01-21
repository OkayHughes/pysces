from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.models_3d.mass_coordinate import create_vertical_grid
from pysces.models_3d.homme.homme_state import get_p_mid
from pysces.models_3d.constants import init_config
from pysces.models_3d.initialization.umjs14 import get_umjs_config
from pysces.models_3d.homme.homme_state import project_model_state, wrap_model_struct
from pysces.models_3d.homme.time_stepping import advance_euler
from pysces.operations_2d.operators import inner_prod
from pysces.models_3d.homme.time_stepping import rfold_state
from pysces.config import device_wrapper, device_unwrapper, jnp, np, wrapper_type, jit
from ..test_init import get_umjs_state
from ..mass_coordinate_grids import cam30
from functools import partial
from ...context import get_figdir
if wrapper_type == "jax":
  from jax.scipy.optimize import minimize
  from jax.scipy.sparse.linalg import gmres
  import jax
  from jaxopt.linear_solve import solve_gmres

sigma_b = 0.70
secpday = 86400
k_a     = 1.0/(40.0)
k_f     = 1.0/(1.0*secpday)
k_s     = 1.0/(4.0*secpday)
dtheta_z= 10.0
dT_y    = 60.0

@jit
def hs_temperature(lat, lon, pi, T, v_grid, config):
  logprat   = jnp.log(pi)-jnp.log(v_grid["reference_pressure"])
  etam      = v_grid["hybrid_a_m"] + v_grid["hybrid_b_m"] 
  pratk     = jnp.exp(config["Rgas"]/config["cp"]*(logprat))
  k_t = (k_a + (k_s-k_a)*(jnp.cos(lat)**2*jnp.cos(lat)**2)[:, :, :, np.newaxis]*
         jnp.maximum(0.0,((etam - sigma_b)/(1.0 - sigma_b))[np.newaxis, np.newaxis, np.newaxis, :]))
  Teq     = jnp.maximum(200.0,(315.0 - dT_y*jnp.sin(lat)[:, :, :, np.newaxis]**2 - dtheta_z*logprat*jnp.cos(lat)[:, :, :, np.newaxis]**2)*pratk)

  hs_T_frc = -k_t *(T-Teq)
  return hs_T_frc, Teq

@jit
def hs_u(u, v_grid):
  etam      = v_grid["hybrid_a_m"] + v_grid["hybrid_b_m"] 
  k_v = k_f*jnp.maximum(0.0,(etam - sigma_b )/(1.0 - sigma_b))
  print(u.shape)
  hs_v_frc = jnp.stack((-k_v[np.newaxis, np.newaxis, np.newaxis, :] * u[:, :, :, :, 0],
                        -k_v[np.newaxis, np.newaxis, np.newaxis, :] * u[:, :, :, :, 1]),
                        axis=-1)
  print(hs_v_frc.shape)
  return hs_v_frc

def test_notopo():
  if wrapper_type != "jax":
    return
  npt = 4
  nx = 10
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  model_config = init_config()

  test_config = get_umjs_config(model_config=model_config)
  test_config["T0P"] = test_config["T0E"]
  model_state, _ = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, mountain=False, hydrostatic=True)
  shapes = [model_state["u"].shape,
           model_state["vtheta_dpi"].shape,
           model_state["dpi"].shape]
  sizes = [model_state["u"].size,
           model_state["vtheta_dpi"].size,
           model_state["dpi"].size]
  model_state["u"] += 10.0 * device_wrapper(np.random.normal(scale=10.0, size=model_state["u"].shape))
  pi = get_p_mid(model_state, v_grid, model_config)
  exner = (pi/model_config["p0"])**(model_config["Rgas"]/model_config["cp"])
  _, T_eq = hs_temperature(h_grid["physical_coords"][:, :, :, 0],
                           h_grid["physical_coords"][:, :, :, 1],
                           pi, jnp.zeros_like(pi),
                           v_grid, model_config)
  model_state["vtheta_dpi"] = (T_eq / exner) * model_state["dpi"]
  model_state = project_model_state(model_state, h_grid, dims, hydrostatic=True)
  @jit
  def flatten(model_state):
    u = model_state["u"].flatten()
    vtheta_dpi = model_state["vtheta_dpi"].flatten()
    dpi = model_state["dpi"].flatten()
    return jnp.concatenate((u, vtheta_dpi, dpi))
  def unpack(model_state_flat, shapes, sizes):
    ct = 0
    u = model_state_flat[:sizes[0]].reshape(shapes[0])
    ct += sizes[0]
    vtheta_dpi = model_state_flat[ct:ct+sizes[1]].reshape(shapes[1])
    ct += sizes[1]
    dpi = model_state_flat[ct:ct+sizes[2]].reshape(shapes[2])
    ct += sizes[2]
    return u, vtheta_dpi, dpi
  @jit
  def advance(model_state_n):
    dt = 100
    print("asdf")
    pi = get_p_mid(model_state_n, v_grid, model_config)
    exner = (pi/model_config["p0"])**(model_config["Rgas"]/model_config["cp"])
    theta = model_state_n["vtheta_dpi"]/model_state_n["dpi"]
    T = theta * exner
    temp_tend, _ = hs_temperature(h_grid["physical_coords"][:, :, :, 0],
                        h_grid["physical_coords"][:, :, :, 1],
                        pi,
                        T,
                        v_grid,
                        model_config)
    theta_dpi_tend = (temp_tend / exner) * model_state_n["dpi"]
    du = hs_u(model_state_n["u"], v_grid)
    model_state_n["u"] += dt * du
    model_state_n["vtheta_dpi"] += dt * theta_dpi_tend
    model_state_np1 = project_model_state(advance_euler(model_state_n, dt, h_grid, v_grid, model_config, dims),
                                        h_grid, dims, hydrostatic=True)
    state_diff = rfold_state(model_state_np1, model_state_n, 1.0, -1.0)
    return state_diff
  def residual(model_state_flat, phi_surf, grad_phi_surf, shapes, sizes):
    dt = 100
    u, vtheta_dpi, dpi = unpack(model_state_flat, shapes, sizes)

    model_state_n = wrap_model_struct(u, vtheta_dpi, dpi, phi_surf, grad_phi_surf, 0.0, 0.0)
    pi = get_p_mid(model_state_n, v_grid, model_config)
    exner = (pi/model_config["p0"])**(model_config["Rgas"]/model_config["cp"])
    theta = model_state_n["vtheta_dpi"]/model_state_n["dpi"]
    T = theta * exner
    temp_tend, _ = hs_temperature(h_grid["physical_coords"][:, :, :, 0],
                        h_grid["physical_coords"][:, :, :, 1],
                        pi,
                        T,
                        v_grid,
                        model_config)
    theta_dpi_tend = (temp_tend / exner) * model_state_n["dpi"]
    du = hs_u(model_state_n["u"], v_grid)
    #model_state_n["u"] += dt * du
    #model_state_n["vtheta_dpi"] += dt * theta_dpi_tend
    model_state_np1 = project_model_state(advance_euler(model_state_n, dt, h_grid, v_grid, model_config, dims),
                                          h_grid, dims, hydrostatic=True)

    state_diff = rfold_state(model_state_np1, model_state, 1.0, -1.0)

    u2d = jnp.sum(state_diff["u"][:, :, :, :, 0], axis=-1)
    v2d = jnp.sum(state_diff["u"][:, :, :, :, 1], axis=-1)
    vtheta_dpi_2d = jnp.sum(state_diff["vtheta_dpi"], axis=-1)
    dpi_2d = jnp.sum(state_diff["dpi"], axis=-1)
    term1 = inner_prod(u2d, u2d, h_grid) 
    term2 = inner_prod(v2d, v2d, h_grid)
    term3 = inner_prod(vtheta_dpi_2d, vtheta_dpi_2d, h_grid)
    term4 = inner_prod(dpi_2d, dpi_2d, h_grid)
    residual = (term1 +
                term2 +
                term3/1e6 +
                term4/10)
    return residual
  def callback(OR):
    print("wheeee")
    print(f"residual: {OR.fun}")
  res_spec = partial(residual, phi_surf=model_state["phi_surf"],
                     grad_phi_surf=model_state["grad_phi_surf"],
                     shapes=shapes,
                     sizes=sizes)
  use_jac = False
  if use_jac:
    print("starting iter")
    advance_spec = partial(advance, v_grid=v_grid)
    for iter in range(2000):
      def apply_jac(vec):
        jac = jax.jvp(advance, [model_state], [vec])[1]
        return jac
      model_state_np1 = advance(model_state)
      print("starting gmres")
      step = solve_gmres(apply_jac, model_state_np1)
      print("after gmres")

      print(device_unwrapper(step["u"]))
      step = wrap_model_struct(step["u"], step["vtheta_dpi"],
                              step["dpi"], model_state["phi_surf"],
                              model_state["grad_phi_surf"],
                              model_state["phi_i"],
                              model_state["w_i"])
      print(step["u"].shape)
      jnp.copy(step)
      print("after copy")
      model_state = rfold_state(model_state, step, 1.0, -1.0)
      print("after rfold")
      res = res_spec(flatten(model_state))
      print(f"iter: {iter}, residual: {res}")
  else:
    print("attempting minimize")
    vec = jax.jvp(res_spec, [flatten(model_state)], [flatten(model_state)])
    step_size = 5e-3
    grad = jax.grad(res_spec, argnums=0)
    for iter in range(3000):
      #minimize(res_spec, flatten(model_state), method="BFGS")
      gradient = grad(flatten(model_state))
      u, vtheta_dpi, dpi = unpack(gradient, shapes, sizes)
      grad_state = wrap_model_struct(u, vtheta_dpi, dpi, model_state["phi_surf"], model_state["grad_phi_surf"], 0.0, 0.0)
      model_state = rfold_state(model_state, grad_state, 1.0, -step_size)
      res = res_spec(flatten(model_state))
      print(f"iter: {iter}, residual: {res}")
    print("asdfadsf")
    import matplotlib.pyplot as plt
    lat = h_grid["physical_coords"][:, :, :, 0].flatten()
    lon = h_grid["physical_coords"][:, :, :, 1].flatten()
    plt.figure()
    plt.tricontourf(lon, lat, model_state["u"][:, :, :, 15, 0].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/u_steady_state.pdf")
