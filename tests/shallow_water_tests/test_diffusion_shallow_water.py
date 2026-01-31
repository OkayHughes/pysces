from pysces.config import jnp, np, device_wrapper
from pysces.shallow_water_models.model_state import wrap_model_state
from pysces.shallow_water_models.constants import init_physics_config_shallow_water
from pysces.shallow_water_models.hyperviscosity import (init_hypervis_config_const,
                                                        init_hypervis_config_tensor,
                                                        eval_hypervis_quasi_uniform,
                                                        eval_hypervis_variable_resolution)
from pysces.shallow_water_models.time_stepping import init_timestep_config
from pysces.shallow_water_models.run_shallow_water import simulate_shallow_water
from pysces.shallow_water_models.galewsky_init import (init_galewsky_config,
                                                       eval_galewsky_wind,
                                                       eval_galewsky_hs,
                                                       eval_galewsky_h)
from pysces.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pysces.horizontal_grid import postprocess_grid


def test_galewsky():
  npt = 4
  for nx in [7, 15, 31]:
    grid, dims = init_quasi_uniform_grid_elem_local(nx, npt)
    grid = postprocess_grid(grid, dims)

    physics_config = init_physics_config_shallow_water()
    test_config = init_galewsky_config(physics_config)

    T = (4 * 3600)  # four days
    dt = 300
    u_init = device_wrapper(eval_galewsky_wind(grid["physical_coords"][:, :, :, 0],
                                               grid["physical_coords"][:, :, :, 1],
                                               test_config))
    h_init = device_wrapper(eval_galewsky_h(grid["physical_coords"][:, :, :, 0],
                                            grid["physical_coords"][:, :, :, 1],
                                            test_config))
    hs_init = device_wrapper(eval_galewsky_hs(grid["physical_coords"][:, :, :, 0],
                                              grid["physical_coords"][:, :, :, 1],
                                              test_config))
    h_init += np.random.normal(scale=3, size=hs_init.shape)
    u_init += np.random.normal(scale=.1, size=u_init.shape)
    init_state = wrap_model_state(u_init, h_init, hs_init)
    diffusion_config_uniform = init_hypervis_config_const(nx, physics_config, nu_div_factor=1.0)
    diffusion_config_variable_res = init_hypervis_config_tensor(grid, dims, physics_config)
    diffusion_tend_uniform = eval_hypervis_quasi_uniform(init_state, grid,
                                                         physics_config, diffusion_config_uniform,
                                                         dims)
    diffusion_tend_variable_res = eval_hypervis_variable_resolution(init_state, grid,
                                                                    physics_config, diffusion_config_variable_res,
                                                                    dims)
    px = 70

    def log_assert(quant1, quant2):
      assert np.abs(np.log10(np.abs(quant1)) - np.log10(np.abs(quant2))) < 1.5

    scale_h_uniform = jnp.percentile(np.abs(diffusion_tend_uniform["h"]), px)
    scale_h_variable = jnp.percentile(np.abs(diffusion_tend_variable_res["h"]), px)
    log_assert(scale_h_uniform, scale_h_variable)
    scale_u_uniform = jnp.percentile(np.abs(diffusion_tend_uniform["u"][:, :, :, 0]), px)
    scale_u_variable = jnp.percentile(np.abs(diffusion_tend_variable_res["u"][:, :, :, 0]), px)
    log_assert(scale_u_uniform, scale_u_variable)
    scale_v_uniform = jnp.percentile(np.abs(diffusion_tend_uniform["u"][:, :, :, 1]), px)
    scale_v_variable = jnp.percentile(np.abs(diffusion_tend_variable_res["u"][:, :, :, 1]), px)
    log_assert(scale_v_uniform, scale_v_variable)
    for diff_config in [diffusion_config_uniform, diffusion_config_variable_res]:
      timestep_config = init_timestep_config(dt, grid, dims, physics_config,
                                             diff_config, sphere=True)
      final_state = simulate_shallow_water(T, init_state, grid,
                                           physics_config, diff_config, timestep_config,
                                           dims, diffusion=True)
      assert not jnp.any(jnp.isnan(final_state["h"]))
      assert not jnp.any(jnp.isnan(final_state["u"]))
