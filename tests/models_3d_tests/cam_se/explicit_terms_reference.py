from pysces.operations_2d.operators import horizontal_gradient, horizontal_vorticity, horizontal_divergence
from pysces.config import np
from pysces.models_3d.cam_se.thermodynamics import sum_species, virtual_temperature, p_mid_moist, p_int_moist, cp_moist, hydrostatic_geopotential
from pysces.models_3d.cam_se.thermodynamics import R_dry as R_dry_fn
from pysces.models_3d.cam_se.thermodynamics import cp_dry as cp_dry_fn
def compute_explicit_terms(u, T, dpi, Q_moist, Q_dry, phi_surf, physics_config, v_grid, h_grid, dims, eta_ave_w=0.0, pgf_formulation=1):

  ptop = v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"] #hvcoord%hyai(1)*hvcoord%ps0
  NELEM = dims["num_elem"]
  npt = dims["npt"]
  NLEV = v_grid["hybrid_a_m"].size
  radius_earth = physics_config["radius_earth"]
  period_earth = physics_config["period_earth"]
  kappa = physics_config["Rgas"]/physics_config["cp"]
  #
  # compute virtual temperature and sum_water
  #
  total_mixing_ratio = sum_species(Q_moist)
  T_v = virtual_temperature(total_mixing_ratio, T)
  R_dry = R_dry_fn(Q_dry, physics_config)
  cp_dry = cp_dry_fn(Q_dry, physics_config)
  dp_dry = np.zeros_like(dpi)
  dp_full = np.zeros_like(dpi)
  for k in range(NLEV):
    dp_dry[:, :,:,k]  = dpi[:, :, :, k]
    dp_full[:, :,:,k] = total_mixing_ratio[:, :, :, k] * dp_dry[:, :, :, k]
  p_int = p_int_moist(dp_full, ptop)
  p_full = p_mid_moist(p_int)
  inv_cp_full = 1.0/cp_moist(Q_moist, physics_config)

  phi = hydrostatic_geopotential(p_full, T_v, R_dry, phi_surf)
  vgrad_p_full = np.zeros_like(dpi)
  vdp_dry = np.zeros_like(u)
  vdp_full = np.zeros_like(u)
  grad_p_full = np.zeros_like(u)
  vtemp = np.zeros_like(u)
  vn0 = np.zeros_like(u)
  pgf_term = np.zeros_like(u)
  vtens = np.zeros_like(u)
  div_dp_dry = np.zeros_like(dpi)
  div_dp_full = np.zeros_like(dpi)
  omega_full = np.zeros_like(dpi)
  omega = np.zeros_like(dpi)
  ttens = np.zeros_like(dpi)
  vort = np.zeros_like(dpi)
  for k in range(NLEV):
    # vertically lagrangian code: we advect dp3d instead of ps
    # we also need grad(p) at all levels (not just grad(ps))
    #p(k)= hyam(k)*ps0 + hybm(k)*ps
    #    = .5_r8*(hyai(k+1)+hyai(k))*ps0 + .5_r8*(hybi(k+1)+hybi(k))*ps
    #    = .5_r8*(ph(k+1) + ph(k) )  = ph(k) + dp(k)/2
    #
    # p(k+1)-p(k) = ph(k+1)-ph(k) + (dp(k+1)-dp(k))/2
    #             = dp(k) + (dp(k+1)-dp(k))/2 = (dp(k+1)+dp(k))/2

    grad_p_full[:, :, :, k, :] = horizontal_gradient(p_full[:, :, :, k], h_grid, a=radius_earth)
    # ==============================
    # compute vgrad_lnps - for omega_full
    # ==============================
    for j in range(npt):
      for i in range(npt):
        v1 = u[:, i, j, k, 0]
        v2 = u[:, i, j, k, 1]
        vgrad_p_full[:, i, j, k] = (v1 * grad_p_full[:, i, j, k, 0] + v2 * grad_p_full[:, i, j, k, 1])
        vdp_dry[:, i, j, k, 0] = v1 * dp_dry[:, i, j, k]
        vdp_dry[:, i, j, k, 1] = v2 * dp_dry[:, i, j, k]
        vdp_full[:, i, j, k, 0] = v1 * dp_full[:, i, j, k]
        vdp_full[:, i, j, k, 1] = v2 * dp_full[:, i, j, k]
    # ================================
    # Accumulate mean Vel_rho flux in vn0
    # ================================
    for j in range(npt):
      for i in range(npt):
        vn0[:, i, j, k, 0] = vn0[:, i, j, k, 0] + eta_ave_w * vdp_dry[:, i, j, k, 0]
        vn0[:, i, j, k, 1] = vn0[:, i, j, k, 1] + eta_ave_w * vdp_dry[:, i, j, k, 1]
    #divdp_dry(:,:,k)
    # =========================================
    #
    # Compute relative vorticity and divergence
    #
    # =========================================
    div_dp_dry[:, :, :, k] = horizontal_divergence(vdp_dry[:, :, :, k, :], h_grid, a=radius_earth)
    div_dp_full[:, :, :, k] = horizontal_divergence(vdp_full[:, :, :, k, :], h_grid, a=radius_earth)
    vort[:, :, :, k] = horizontal_vorticity(u[:, :, :, k, :], h_grid, a=radius_earth)

  # ====================================================
  # Compute omega_full
  # ====================================================
  ckk = 0.5
  suml = np.zeros((NELEM, npt, npt))
  for k in range(NLEV):
    for j in range(npt):
      for i in range(npt):
        term = -div_dp_full[:, i, j, k]

        v1 = u[:, i, j, k, 0]
        v2 = u[:, i, j, k, 1]

        omega_full[:, i, j, k] = suml[:, i, j] + ckk * term + vgrad_p_full[:, i, j, k]
        suml[:, i, j] = suml[:, i, j] + term

  for k in range(NLEV):
    omega[:, :, :, k] = omega[:, :, :, k] + eta_ave_w * omega_full[:, :, :, k]
  # ==============================================
  # Compute phi + kinetic energy term: 10*nv*nv Flops
  # ==============================================
  for k in range(NLEV):
    Ephi = np.zeros((NELEM, npt, npt))
    for j in range(npt):
      for i in range(npt):
        v1     = u[:, i, j, k, 0]
        v2     = u[:, i, j, k, 1]
        E = 0.5 * (v1 * v1 + v2 * v2)
        Ephi[:, i, j] = E + phi[:, i, j, k]
    # ================================================
    # compute gradp term (ps/p)*(dp/dps)*T
    # ================================================
    vtemp = horizontal_gradient(T[:, :, :, k], h_grid, a=radius_earth)
    vgrad_T = np.zeros((NELEM, npt, npt))
    for j in range(npt):
      for i in range(npt):
        v1     = u[:, i, j, k, 0]
        v2     = u[:, i, j, k, 1]
        vgrad_T[:, i, j] = v1 * vtemp[:, i, j, 0] + v2 * vtemp[:, i, j, 1]


    # vtemp = grad ( E + PHI )
    # vtemp = gradient_sphere(Ephi(:,:),deriv,elem(ie)%Dinv)
    vtemp = horizontal_gradient(Ephi[:, :, :], h_grid, a=radius_earth)
    density_inv = R_dry[:, :, :, k] * T_v[:, :, :, k] / p_full[:, :, :, k]
    if (pgf_formulation == 1 or (pgf_formulation == 3 and v_grid["hybrid_b_m"][k] > 1e-9)):
      exner = (p_full[:, :, :, k] / v_grid["reference_pressure"])**kappa
      theta_v = T_v[:, :, :, k] / exner
      grad_exner = horizontal_gradient(exner, h_grid, a=radius_earth)
      pgf_term[:, :, :, 0] = cp_dry[:, :, :, k] * theta_v * grad_exner[:, :, :, 0]
      pgf_term[:, :, :, 1] = cp_dry[:, :, :, k] * theta_v * grad_exner[:, :, :, 1]
      # balanced ref profile correction:
      # reference temperature profile (Simmons and Jiabin, 1991, QJRMS, Section 2a)
      #
      #  Tref = T0+T1*Exner
      #  T1 = .0065*Tref*Cp/g ! = ~191
      #  T0 = Tref-T1         ! = ~97
      #
      lapse_rate = physics_config["reference_profiles"]["T_ref_lapse_rate"]
      T_ref = physics_config["reference_profiles"]["T_ref"]
      T1 = (lapse_rate * T_ref * physics_config["cp"] / physics_config["gravity"])
      T0 = T_ref - T1
      if v_grid["hybrid_b_m"][k] > 1e-9:
        # only apply away from constant pressure levels
        grad_logexner = horizontal_gradient(np.log(exner), h_grid, a=radius_earth)
        pgf_term[:, :, :, 0] = pgf_term[:, :, :, 0] + cp_dry[:, :, :, k] * T0 * (grad_logexner[:, :, :, 0] - grad_exner[:, :, :, 0] / exner)
        pgf_term[:, :, :, 1] = pgf_term[:, :, :, 1] + cp_dry[:, :, :, k] * T0 * (grad_logexner[:, :, :, 1] - grad_exner[:, :, :, 1] / exner)
    elif pgf_formulation == 2 or pgf_formulation == 3:
      pgf_term[:, :, :, 0]  = density_inv * grad_p_full[:, :, :, k, 0]
      pgf_term[:, :, :, 1]  = density_inv * grad_p_full[:, :, :, k, 1]
    else:
      raise ValueError("Pressure gradient not implemented")
    for j in range(npt):
      for i in range(npt):
        glnps1 = pgf_term[:, i, j, 0]
        glnps2 = pgf_term[:, i, j, 1]
        v1 = u[:, i, j, k, 0]
        v2 = u[:,i,j,k,1]
        f_cor = 2 * period_earth * np.sin(h_grid["physical_coords"][:, i, j, 0])
        vtens[:, i, j, k, 0] =  v2 * (f_cor[:, i, j] + vort[:, i, j, k]) - vtemp[:, i, j, 0] - glnps1
        vtens[:, i, j, k, 1] = -v1 * (f_cor[:, i, j] + vort[:, i, j, k]) - vtemp[:, i, j, 1] - glnps2
        ttens[:, i, j, k]  =  -vgrad_T[:, i, j] + density_inv[:, i, j] * omega_full[:, i, j, k] * inv_cp_full[:, i, j, k]
  dptens = -div_dp_dry
  return {"u": vtens,
          "T": ttens,
          "dpi": dptens}

