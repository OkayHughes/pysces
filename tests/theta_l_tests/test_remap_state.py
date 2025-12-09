from spherical_spectral_element.theta_l.initialization.umjs14 import (get_umjs_config,
                                                                      evaluate_pressure_temperature)
from spherical_spectral_element.theta_l.constants import init_config
from spherical_spectral_element.equiangular_metric import create_quasi_uniform_grid
from spherical_spectral_element.config import jnp, np, jax_wrapper
from .vertical_grids import cam30
from spherical_spectral_element.theta_l.infra import interface_to_model, g_from_phi
from spherical_spectral_element.theta_l.vertical_coordinate import create_vertical_grid
from .test_init import get_umjs_state
from spherical_spectral_element.theta_l.model_state import remap_state

def test_eos_nonhydro():
  nx = 5
  h_grid, dims = create_quasi_uniform_grid(nx)
  lat = h_grid["physical_coords"][:, :, :, 0]
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  for mountain in [False]:
    model_config = init_config()
    test_config = get_umjs_config(model_config=model_config)
    lat = h_grid["physical_coords"][:, :, :, 0]
    model_state, _ = get_umjs_state(h_grid, v_grid, model_config, test_config, dims,
                                    mountain=mountain, hydrostatic=False)
    u = model_state["u"]
    w_i = np.random.normal(size=model_state["w_i"].shape)
    w_i[:, :, :, -1] = (u[:, :, :, -1, 0] * model_state["grad_phi_surf"][:, :, :, 0] +
                        u[:, :, :, -1, 1] * model_state["grad_phi_surf"][:, :, :, 1]) / g_from_phi(model_state["phi_surf"], model_config, deep=False)
    model_state["w_i"] = jax_wrapper(w_i)
    model_state_remapped = remap_state(model_state, v_grid, model_config, len(v_grid["hybrid_a_m"]), hydrostatic=False, deep=False)



    for field in ["u", "vtheta_dpi",
                  "dpi", "phi_surf",
                  "grad_phi_surf", "phi_i",
                  "w_i"]:
      assert(jnp.max(jnp.abs(model_state[field] - model_state_remapped[field])) < 1e-5)