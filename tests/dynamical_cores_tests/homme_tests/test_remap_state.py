from pysces.analytic_initialization.moist_baroclinic_wave import get_umjs_config
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.config import jnp, np, device_wrapper
from ..mass_coordinate_grids import cam30
from pysces.dynamical_cores.utils_3d import g_from_phi
from pysces.dynamical_cores.mass_coordinate import create_vertical_grid
from ..test_init import get_umjs_state
from pysces.dynamical_cores.model_state import remap_dynamics
from pysces.model_info import models


def test_remap_state():
  npt = 4
  nx = 5
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  model = models.homme_nonhydrostatic
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
  for mountain in [False]:
    model_config = init_physics_config(model)
    test_config = get_umjs_config(model_config=model_config)
    model_state = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, model,
                                 mountain=mountain)
    dynamics = model_state["dynamics"]
    static_forcing = model_state["static_forcing"]
    u = dynamics["u"]
    w_i = np.random.normal(size=dynamics["w_i"].shape)
    w_i[:, :, :, -1] = ((u[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
                        u[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
                        g_from_phi(static_forcing["phi_surf"], model_config, model))
    dynamics["w_i"] = device_wrapper(w_i)
    dynamics_remapped = remap_dynamics(dynamics, static_forcing, v_grid, model_config, len(v_grid["hybrid_a_m"]), model)

    for field in ["u", "theta_v_d_mass",
                  "d_mass", "phi_i",
                  "w_i"]:
      assert(jnp.max(jnp.abs(dynamics[field] - dynamics_remapped[field])) < 1e-5)
