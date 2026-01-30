from pysces.model_info import models, cam_se_models, hydrostatic_models, homme_models, thermodynamic_variable_names
from pysces.config import jnp, device_wrapper, device_unwrapper, np
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.analytic_initialization.moist_baroclinic_wave import get_umjs_config
from pysces.operations_2d.local_assembly import project_scalar
from pysces.operations_2d.operators import horizontal_weak_laplacian, horizontal_gradient, inner_product
from pysces.mesh_generation.element_local_metric import create_quasi_uniform_grid_elem_local, create_mobius_like_grid_elem_local
from pysces.dynamical_cores.mass_coordinate import create_vertical_grid
from pysces.dynamical_cores.hyperviscosity import vector_harmonic_3d, scalar_harmonic_3d, init_hypervis_config_const, get_ref_states, calc_hypervis_harmonic, sponge_layer, get_nu_ramp, hypervis_terms, init_hypervis_config_tensor
from pysces.dynamical_cores.model_state import project_scalar_3d, project_dynamics_state, sum_dynamics_states, advance_dynamics, wrap_dynamics_struct
from pysces.dynamical_cores.utils_3d import sphere_dot
from pysces.dynamical_cores.time_stepping import init_timestep_config
from .test_init import get_umjs_state
from .mass_coordinate_grids import cam30, vertical_grid_finite_diff
from ..context import get_figdir


def analytic_sph_harm(h_grid, physics_config):
  m = 1
  l = 2
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  norm_const = jnp.sqrt(15.0 / (8.0 * jnp.pi)) 
  Ymn = -norm_const * jnp.sin(lat) * jnp.cos(lat) * jnp.cos(lon)
  eigval = -l * (l + 1) / physics_config["radius_earth"]**2
  curl_Ymn_vec = norm_const * jnp.stack((jnp.cos(lon) * jnp.cos(2*lat),
                                         jnp.sin(lon) * jnp.sin(lat)), axis=-1)
  return eigval, Ymn, curl_Ymn_vec


def test_basic_operators():
  npt = 4
  nx = 15
  model = models.cam_se
  physics_config = init_physics_config(model, radius_earth=2.0)
  radius_earth = physics_config["radius_earth"]
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  eigval, Ymn, curl_Ymn_vec = analytic_sph_harm(h_grid, physics_config)
  laplace_Ymn_discont = horizontal_weak_laplacian(Ymn, h_grid, a=radius_earth)
  laplace_Ymn = project_scalar(laplace_Ymn_discont, h_grid, dims)
  # check that we can resolve our spherical harmonic.
  eps = 1e-2
  diff = laplace_Ymn - eigval * Ymn
  assert jnp.max(jnp.abs(diff)) < eps
  test_scalar = Ymn[:, :, :, jnp.newaxis]
  # use curl form vector spherical harmonics
  test_vector = curl_Ymn_vec[:, :, :, jnp.newaxis, :]
  lap_scalar = scalar_harmonic_3d(test_scalar, h_grid, physics_config)
  lap_scalar_cont = project_scalar_3d(lap_scalar, h_grid, dims)
  assert jnp.max(jnp.abs(lap_scalar_cont.squeeze() - eigval * Ymn)) < eps
  lap_vector = vector_harmonic_3d(test_vector, h_grid, physics_config, 1.0)
  lap_vector_cont = jnp.stack((project_scalar_3d(lap_vector[:, :, :, :, 0], h_grid, dims),
                               project_scalar_3d(lap_vector[:, :, :, :, 1], h_grid, dims)), axis=-1)
  analytic_result = eigval * test_vector
  assert jnp.max(jnp.abs(lap_vector_cont.squeeze() - (analytic_result.squeeze()))) < .05


def test_quasi_uniform():
  npt = 4
  nx = 15
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  v_grid_tmp = vertical_grid_finite_diff(6)
  for model in [models.homme_hydrostatic, models.homme_nonhydrostatic, models.cam_se,]:
    v_grid = create_vertical_grid(v_grid_tmp["hybrid_a_i"],
                                  v_grid_tmp["hybrid_b_i"],
                                  v_grid_tmp["p0"],
                                  model)
    physics_config = init_physics_config(model, radius_earth=2)
    diffusion_config = init_hypervis_config_const(nx, physics_config, v_grid, nu_div_factor=1.0)
    eigval, Ymn, curl_Ymn_vec = analytic_sph_harm(h_grid, physics_config)
    test_config = get_umjs_config(model_config=physics_config)
    model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, eps=1e-3)
    dynamics = model_state["dynamics"]
    dynamics["u"] = curl_Ymn_vec[:, :, :, jnp.newaxis, :] * jnp.ones_like(dynamics["u"])
    dynamics["d_mass"] = Ymn[:, :, :, jnp.newaxis] * jnp.ones_like(dynamics["d_mass"])
    if model in cam_se_models:
      dynamics["T"] = Ymn[:, :, :, jnp.newaxis] * jnp.ones_like(dynamics["T"])
    elif model in homme_models:
      dynamics["theta_v_d_mass"] = jnp.ones_like(dynamics["theta_v_d_mass"]) * Ymn[:, :, :, jnp.newaxis]
    if model not in hydrostatic_models:
      dynamics["phi_i"] = Ymn[:, :, :, jnp.newaxis] * jnp.ones_like(dynamics["phi_i"])
      dynamics["w_i"] = Ymn[:, :, :, jnp.newaxis] * jnp.ones_like(dynamics["w_i"])
    # test hyperviscosity
    for apply_nu in [True, False]:
      laplace_dynamics_discont = calc_hypervis_harmonic(dynamics, h_grid, physics_config, diffusion_config, model, apply_nu=apply_nu)
      laplace_dynamics = project_dynamics_state(laplace_dynamics_discont, h_grid, dims, model)
      eps = 1e-2
      nus = {}
      nus["u"] = diffusion_config["nu"] if apply_nu else 1.0
      nus["phi_i"]= diffusion_config["nu_phi"] if apply_nu else 1.0
      nus["d_mass"]= diffusion_config["nu_d_mass"] if apply_nu else 1.0
      nus["thermo"] = diffusion_config["nu"] if apply_nu else 1.0
      nus["w_i"] = diffusion_config["nu"] if apply_nu else 1.0

      assert jnp.max(jnp.abs(nus["u"] * eigval * dynamics["u"] - laplace_dynamics["u"])) / nus["u"] < eps
      assert jnp.max(jnp.abs(nus["d_mass"] * eigval * dynamics["d_mass"] - laplace_dynamics["d_mass"])) / nus["d_mass"] < eps
      if model in cam_se_models:
        assert jnp.max(jnp.abs(nus["thermo"] * eigval * dynamics["T"] - laplace_dynamics["T"])) / nus["thermo"] < eps
      elif model in homme_models:
        assert jnp.max(jnp.abs((nus["thermo"] * eigval * dynamics["theta_v_d_mass"] - laplace_dynamics["theta_v_d_mass"]))) / nus["thermo"] < eps
      if model not in hydrostatic_models:
        assert jnp.max(jnp.abs(nus["phi_i"] * eigval * dynamics["phi_i"] - laplace_dynamics["phi_i"])) / nus["phi_i"] < eps
        assert jnp.max(jnp.abs(nus["w_i"] * eigval * dynamics["w_i"] -  laplace_dynamics["w_i"])) / nus["w_i"] < eps


def test_nu_ramp():
  model = models.homme_hydrostatic
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
  nu_ramp = get_nu_ramp(v_grid, 5).squeeze()
  for lev_idx in range(nu_ramp.size):
    assert nu_ramp[lev_idx] > 0
    assert nu_ramp[lev_idx] <= 8.0
    if lev_idx > 0:
      assert nu_ramp[lev_idx] < nu_ramp[lev_idx-1]

  

def test_sponge_layer_energy_estimate():
  npt = 4
  nx = 15
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  v_grid_tmp = vertical_grid_finite_diff(6)
  for model in [models.homme_hydrostatic, models.homme_nonhydrostatic, models.cam_se,]:
    v_grid = create_vertical_grid(v_grid_tmp["hybrid_a_i"],
                                  v_grid_tmp["hybrid_b_i"],
                                  v_grid_tmp["p0"],
                                  model)
    physics_config = init_physics_config(model)
    diffusion_config = init_hypervis_config_const(nx, physics_config, v_grid, nu_div_factor=1.0)
    timestep_config = init_timestep_config(1000, h_grid, physics_config, diffusion_config, dims, model)
    test_config = get_umjs_config(model_config=physics_config, T0E=310, T0P=310)
    model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, eps=1e-3)
    dynamics = model_state["dynamics"]
    def pert(vals, scale=1.0):
      return device_wrapper(np.random.normal(size=vals.shape, scale=scale))
    dynamics["u"] += pert(dynamics["u"])
    dynamics["d_mass"] += pert(dynamics["d_mass"], scale=10)
    if model in cam_se_models:
      dynamics["T"] += pert(dynamics["T"])
    elif model in homme_models:
      dynamics["theta_v_d_mass"] += pert(dynamics["theta_v_d_mass"], scale=10) * dynamics["d_mass"]
    if model not in hydrostatic_models:
      dynamics["phi_i"] += pert(dynamics["phi_i"])
      dynamics["w_i"] += pert(dynamics["w_i"])
    dt = timestep_config["sponge"]["dt"]
    n_sponge = diffusion_config["nu_ramp"].size
    dynamics_new = sponge_layer(dynamics, dt, h_grid, physics_config, diffusion_config, dims, model)
    fields = ["u", "d_mass", thermodynamic_variable_names[model]]
    if model not in hydrostatic_models:
      fields += ["phi_i", "w_i"]
    prev_vals = {}
    for field in fields:
      prev_vals[field] = [1e6 for _ in range(n_sponge)]
    for iter_idx in range(10):
      dynamics_new = sponge_layer(dynamics_new, dt, h_grid, physics_config, diffusion_config, dims, model)
      for k_idx in range(n_sponge):
        for field in fields:
          if field == "u":
            vals = 0.5 * (dynamics_new["u"][:, :, :, k_idx, 0]**2 + dynamics_new["u"][:, :, :, k_idx, 1]**2)
          else:
            vals = dynamics_new[field][:, :, :, k_idx]
          total = inner_product(vals, vals, h_grid)
          ratio = total / prev_vals[field][k_idx]
          if iter_idx > 0:
            assert ratio < 1.0
          prev_vals[field][k_idx] = total
        
def test_hypervis_energy_estimate_quasi_uniform():
  # when stability condition is violated, 
  # energy estimate for L2 norm of solution 
  # does not hold. E.g., multiply dt times 2 
  # and this test fails.
  npt = 4
  nx = 15
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  v_grid_tmp = vertical_grid_finite_diff(6)
  for model in [models.homme_hydrostatic, models.homme_nonhydrostatic, models.cam_se]:
    v_grid = create_vertical_grid(v_grid_tmp["hybrid_a_i"],
                                  v_grid_tmp["hybrid_b_i"],
                                  v_grid_tmp["p0"],
                                  model)
    nlev = v_grid["hybrid_a_m"].size
    physics_config = init_physics_config(model)
    diffusion_config_constant = init_hypervis_config_const(nx, physics_config, v_grid, nu_div_factor=1.0)
    diffusion_config_tensor = init_hypervis_config_tensor(h_grid, v_grid, dims, physics_config)
    for diffusion_config in [diffusion_config_constant, diffusion_config_tensor]:
      timestep_config = init_timestep_config(1000, h_grid, physics_config, diffusion_config, dims, model)
      test_config = get_umjs_config(model_config=physics_config, T0E=310, T0P=310)
      model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, eps=1e-3)
      dynamics = model_state["dynamics"]
      if model not in hydrostatic_models:
        phi_i = jnp.copy(dynamics["phi_i"])
        w_i = jnp.copy(dynamics["w_i"])
      else:
        phi_i = None
        w_i = None
      dynamics_base = wrap_dynamics_struct(jnp.copy(dynamics["u"]),
                                          jnp.copy(dynamics[thermodynamic_variable_names[model]]),
                                          jnp.copy(dynamics["d_mass"]),
                                          model,
                                          phi_i=phi_i,
                                          w_i=w_i)
      def pert(vals, scale=1.0):
        return device_wrapper(np.random.normal(size=vals.shape, scale=scale))
      dynamics["u"] += pert(dynamics["u"])
      dynamics["d_mass"] += pert(dynamics["d_mass"], scale=10)
      if model in cam_se_models:
        dynamics["T"] += pert(dynamics["T"])
      elif model in homme_models:
        dynamics["theta_v_d_mass"] += pert(dynamics["theta_v_d_mass"], scale=10) * dynamics["d_mass"]
      if model not in hydrostatic_models:
        dynamics["phi_i"] += pert(dynamics["phi_i"])
        dynamics["w_i"] += pert(dynamics["w_i"])
      ref_state = get_ref_states(model_state["static_forcing"]["phi_surf"], v_grid, physics_config, diffusion_config, model)
      dt = timestep_config["hyperviscosity"]["dt"]
      dynamics_step = hypervis_terms(dynamics, model_state["static_forcing"], h_grid, v_grid, dims, physics_config, diffusion_config, model)
      dynamics_new = advance_dynamics([dynamics, dynamics_step], [1.0, dt], model)
      fields = ["u", "d_mass", thermodynamic_variable_names[model]]
      if model not in hydrostatic_models:
        fields += ["phi_i", "w_i"]
      prev_vals = {}
      for field in fields:
        prev_vals[field] = [1e6 for _ in range(nlev)]
      for iter_idx in range(10):
        dynamics_step = hypervis_terms(dynamics_new, model_state["static_forcing"], h_grid, v_grid, dims, physics_config, diffusion_config, model)
        dynamics_new = advance_dynamics([dynamics_new, dynamics_step], [1.0, dt], model)
        for k_idx in range(nlev):
          for field in fields:
            if field == "u":
              u_pert = dynamics_new["u"][:, :, :, k_idx, :] - dynamics_base["u"][:, :, :, k_idx, :]
              vals = 0.5 * (u_pert[:, :, :, 0]**2 + u_pert[:, :, :, 1]**2)
            elif field == "theta_v_d_mass":
              theta_v_pert = dynamics_new["theta_v_d_mass"][:, :, :, k_idx] 
              vals = theta_v_pert / dynamics["d_mass"][:, :, :, k_idx] - ref_state["theta_v"][:, :, :, k_idx]
            elif field == "T":
              vals = dynamics_new["T"][:, :, :, k_idx] - ref_state["T"][:, :, :, k_idx]
            else:
              vals = dynamics_new[field][:, :, :, k_idx] - dynamics_base[field][:, :, :, k_idx]
            total = inner_product(vals, vals, h_grid)
            ratio = total / prev_vals[field][k_idx]
            if iter_idx > 0:
              assert ratio < 1.0
            prev_vals[field][k_idx] = total


def test_hypervis_energy_estimate_mobius():
  # when stability condition is violated, 
  # energy estimate for L2 norm of solution 
  # does not hold. E.g., multiply dt times 2 
  # and this test fails.
  for _ in range(3):
    scale = device_wrapper(np.random.uniform(high=1.2, low=0.8, size=3))
    offset = device_wrapper(np.random.uniform(high=0.2, low=0.2, size=3))
    model = models.homme_nonhydrostatic
    npt = 4
    nx = 15
    h_grid, dims = create_mobius_like_grid_elem_local(nx, npt, axis_dilation=scale, offset=offset)
    lat = h_grid["physical_coords"][:, :, :, 0]
    lon = h_grid["physical_coords"][:, :, :, 1]
    v_grid_tmp = vertical_grid_finite_diff(6)
    v_grid = create_vertical_grid(v_grid_tmp["hybrid_a_i"],
                                  v_grid_tmp["hybrid_b_i"],
                                  v_grid_tmp["p0"],
                                  model)
    nlev = v_grid["hybrid_a_m"].size
    physics_config = init_physics_config(model)
    diffusion_config = init_hypervis_config_tensor(h_grid, v_grid, dims, physics_config)
    timestep_config = init_timestep_config(1000, h_grid, physics_config, diffusion_config, dims, model)
    test_config = get_umjs_config(model_config=physics_config, T0E=310, T0P=310)
    model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, eps=1e-3)
    dynamics = model_state["dynamics"]
    if model not in hydrostatic_models:
      phi_i = jnp.copy(dynamics["phi_i"])
      w_i = jnp.copy(dynamics["w_i"])
    else:
      phi_i = None
      w_i = None
    dynamics_base = wrap_dynamics_struct(jnp.copy(dynamics["u"]),
                                        jnp.copy(dynamics[thermodynamic_variable_names[model]]),
                                        jnp.copy(dynamics["d_mass"]),
                                        model,
                                        phi_i=phi_i,
                                        w_i=w_i)
    def pert(vals, scale=1.0):
      return device_wrapper(np.random.normal(size=vals.shape, scale=scale))
    dynamics["u"] += pert(dynamics["u"])
    dynamics["d_mass"] += pert(dynamics["d_mass"], scale=10)
    if model in cam_se_models:
      dynamics["T"] += pert(dynamics["T"])
    elif model in homme_models:
      dynamics["theta_v_d_mass"] += pert(dynamics["theta_v_d_mass"], scale=10) * dynamics["d_mass"]
    if model not in hydrostatic_models:
      dynamics["phi_i"] += pert(dynamics["phi_i"])
      dynamics["w_i"] += pert(dynamics["w_i"])
    ref_state = get_ref_states(model_state["static_forcing"]["phi_surf"], v_grid, physics_config, diffusion_config, model)
    dt = timestep_config["hyperviscosity"]["dt"]
    dynamics_step = hypervis_terms(dynamics, model_state["static_forcing"], h_grid, v_grid, dims, physics_config, diffusion_config, model)
    dynamics_new = advance_dynamics([dynamics, dynamics_step], [1.0, dt], model)
    fields = ["u", "d_mass", thermodynamic_variable_names[model]]
    if model not in hydrostatic_models:
      fields += ["phi_i", "w_i"]
    prev_vals = {}
    for field in fields:
      prev_vals[field] = [1e6 for _ in range(nlev)]
    for iter_idx in range(10):
      dynamics_step = hypervis_terms(dynamics_new, model_state["static_forcing"], h_grid, v_grid, dims, physics_config, diffusion_config, model)
      dynamics_new = advance_dynamics([dynamics_new, dynamics_step], [1.0, dt], model)
      for k_idx in range(nlev):
        for field in fields:
          if field == "u":
            u_pert = dynamics_new["u"][:, :, :, k_idx, :] - dynamics_base["u"][:, :, :, k_idx, :]
            vals = 0.5 * (u_pert[:, :, :, 0]**2 + u_pert[:, :, :, 1]**2)
          elif field == "theta_v_d_mass":
            theta_v_pert = dynamics_new["theta_v_d_mass"][:, :, :, k_idx] 
            vals = theta_v_pert / dynamics["d_mass"][:, :, :, k_idx] - ref_state["theta_v"][:, :, :, k_idx]
          elif field == "T":
            vals = dynamics_new["T"][:, :, :, k_idx] - ref_state["T"][:, :, :, k_idx]
          else:
            vals = dynamics_new[field][:, :, :, k_idx] - dynamics_base[field][:, :, :, k_idx]
          total = inner_product(vals, vals, h_grid)
          ratio = total / prev_vals[field][k_idx]
          if iter_idx > 0:
            assert ratio < 1.0
          prev_vals[field][k_idx] = total
