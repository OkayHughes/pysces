from pysces.models_3d.initialization.umjs14 import get_umjs_config
from pysces.models_3d.constants import init_config
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.config import jnp, np, device_wrapper
from ..mass_coordinate_grids import cam30
from pysces.models_3d.utils_3d import g_from_phi
from pysces.models_3d.mass_coordinate import create_vertical_grid
from ..test_init import get_umjs_state
from pysces.models_3d.homme.homme_state import remap_state


def test_remap_state():
  npt = 4
  nx = 5
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  for mountain in [False]:
    model_config = init_config()
    test_config = get_umjs_config(model_config=model_config)
    model_state, _ = get_umjs_state(h_grid, v_grid, model_config, test_config, dims,
                                    mountain=mountain, hydrostatic=False)
    u = model_state["u"]
    w_i = np.random.normal(size=model_state["w_i"].shape)
    w_i[:, :, :, -1] = ((u[:, :, :, -1, 0] * model_state["grad_phi_surf"][:, :, :, 0] +
                        u[:, :, :, -1, 1] * model_state["grad_phi_surf"][:, :, :, 1]) /
                        g_from_phi(model_state["phi_surf"], model_config, deep=False))
    model_state["w_i"] = device_wrapper(w_i)
    model_state_remapped = remap_state(model_state, v_grid, model_config, len(v_grid["hybrid_a_m"]),
                                       hydrostatic=False, deep=False)

    for field in ["u", "theta_v_d_mass",
                  "d_mass", "phi_surf",
                  "grad_phi_surf", "phi_i",
                  "w_i"]:
      assert(jnp.max(jnp.abs(model_state[field] - model_state_remapped[field])) < 1e-5)
