from .test_init import get_umjs_state
from spherical_spectral_element.equiangular_metric import create_quasi_uniform_grid
from .vertical_grids import cam30
from spherical_spectral_element.theta_l.model_state import dss_model_state
from spherical_spectral_element.theta_l.vertical_coordinate import create_vertical_grid
from spherical_spectral_element.theta_l.initialization.umjs14 import get_umjs_config
from spherical_spectral_element.config import jnp, device_wrapper, np
from spherical_spectral_element.theta_l.constants import init_config
from spherical_spectral_element.theta_l.explicit_terms import calc_energy_quantities
from spherical_spectral_element.operators import inner_prod
from spherical_spectral_element.assembly import dss_scalar


def test_notopo():
  nx = 5
  h_grid, dims = create_quasi_uniform_grid(nx)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  model_config = init_config()
  test_config = get_umjs_config(model_config=model_config)
  for _ in range(10):
    print("\nInitializing random atmosphere\n" + "=" * 28 + "\n")
    model_state, _ = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, mountain=False, hydrostatic=False)
    model_state["u"] += device_wrapper(np.random.normal(scale=0.1, size=model_state["u"].shape))
    model_state["w_i"] += device_wrapper(np.random.normal(scale=0.1, size=model_state["w_i"].shape))
    phi_pert = jnp.concatenate((device_wrapper(np.random.normal(scale=1.0,
                                                                size=model_state["phi_i"][:, :, :, :-1].shape)),
                                jnp.zeros_like(model_state["phi_surf"])[:, :, :, np.newaxis]), axis=-1)
    model_state["phi_i"] = model_state["phi_i"] + phi_pert
    model_state = dss_model_state(model_state, h_grid, dims, hydrostatic=False)
    pairs, empirical_tendencies = calc_energy_quantities(model_state, h_grid, v_grid, model_config, dims)

    def compare_quantities(a, b, label):
      a_int = inner_prod(dss_scalar(a, h_grid, dims), jnp.ones_like(a), h_grid)
      b_int = inner_prod(dss_scalar(b, h_grid, dims), jnp.ones_like(b), h_grid)
      c_int = inner_prod(dss_scalar(a + b, h_grid, dims), jnp.ones_like(a), h_grid)
      print(f"pair {label}: a_int: {a_int}, b_int: {b_int}, sum: {c_int}")
      assert (np.abs(c_int) < 1e-8)
    for pair_name in ["ke_ke_1",
                      "ke_ke_2",
                      "ke_ke_3",
                      "ke_ke_4",
                      "ke_ke_5",
                      "ke_ke_6",
                      "ke_pe_1",
                      "pe_pe_1",
                      "ke_ie_1",
                      "ke_ie_2",
                      "ke_ie_3"]:
      compare_quantities(*pairs[pair_name], pair_name)
    total_energy_change = inner_prod(dss_scalar(empirical_tendencies["ke"] +
                                                empirical_tendencies["ie"] +
                                                empirical_tendencies["pe"], h_grid, dims),
                                     jnp.ones_like(empirical_tendencies["ke"]),
                                     h_grid)
    print(f"total energy change: {total_energy_change}")
    assert (np.abs(total_energy_change))
