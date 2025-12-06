from ..config import jnp, vmap_1d_apply
from ..assembly import dss_scalar, dss_scalar_for
from ..operators import sphere_gradient


def wrap_model_struct(u, vtheta_dpi, dpi, phi_surf, grad_phi_surf, phi_i, w_i):
  state = {"u": u,
           "vtheta_dpi": vtheta_dpi,
           "dpi": dpi,
           "phi_surf": phi_surf,
           "grad_phi_surf": grad_phi_surf,
           "phi_i": phi_i,
           "w_i": w_i
           }
  return state

def wrap_tracer_avg_struct(avg_u, avg_dpi, avg_dpi_dissip):
  return {"avg_v": avg_u,
          "avg_dpi": avg_dpi,
          "avg_dpi_dissip": avg_dpi_dissip}

def init_model_struct(u, vtheta_dpi, dpi, phi_surf, phi_i, w_i, h_grid, dims, config):
  grad_phi_surf_discont = sphere_gradient(phi_surf, h_grid, a=config["radius_earth"])
  grad_phi_surf = jnp.stack((dss_scalar(grad_phi_surf_discont[:, :, :, 0], h_grid, dims),
                             dss_scalar(grad_phi_surf_discont[:, :, :, 1], h_grid, dims)), axis=-1)
  state = {"u": u,
           "vtheta_dpi": vtheta_dpi,
           "dpi": dpi,
           "phi_surf": phi_surf,
           "grad_phi_surf": grad_phi_surf,
           "phi_i": phi_i,
           "w_i": w_i
           }
  return state


def init_tracer_struct(Q):
  return {"Q": Q}


def dss_scalar_3d(variable, h_grid, dims, scaled=True):
  def dss_onlyarg(vec):
    return dss_scalar(vec, h_grid, dims, scaled=scaled)
  return vmap_1d_apply(dss_onlyarg, variable, -1, -1)


def dss_scalar_3d_for(variable, h_grid, dims, scaled=True):
  levs = []
  for lev_idx in range(variable.shape[-1]):
    levs.append(dss_scalar_for(variable[:, :, :, lev_idx], h_grid))
  return jnp.stack(levs, axis=-1)


def dss_model_state(state_in, h_grid, dims, scaled=True, hydrostatic=True):
  u_dss = dss_scalar_3d(state_in["u"][:, :, :, :, 0], h_grid, dims, scaled=scaled)
  v_dss = dss_scalar_3d(state_in["u"][:, :, :, :, 1], h_grid, dims, scaled=scaled)
  vtheta_dpi_dss = dss_scalar_3d(state_in["vtheta_dpi"][:, :, :, :], h_grid, dims, scaled=scaled)
  dpi_dss = dss_scalar_3d(state_in["dpi"][:, :, :, :], h_grid, dims, scaled=scaled)
  if hydrostatic:
    w_i_dss = state_in["w_i"]
    phi_i_dss = state_in["phi_i"]
  else:
    w_i_dss = dss_scalar_3d(state_in["w_i"], h_grid, dims, scaled=scaled)
    phi_i_dss = dss_scalar_3d(state_in["phi_i"], h_grid, dims, scaled=scaled)
  return wrap_model_struct(jnp.stack((u_dss, v_dss), axis=-1),
                           vtheta_dpi_dss, dpi_dss, state_in["phi_surf"],
                           state_in["grad_phi_surf"],
                           phi_i_dss,
                           w_i_dss)
