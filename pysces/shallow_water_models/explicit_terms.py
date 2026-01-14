from ..config import jit, jnp, np
from ..operations_2d.operators import manifold_vorticity, manifold_gradient, manifold_divergence
from .model_state import create_state_struct


@jit
def calc_rhs(state_in, grid, physics_config):
  """
  Calculate the explicit right-hand-side of the rotating shallow water equations on a sphere.
  
  Parameters
  ----------
  state_in : `ShallowWaterModelState`
      the 1st param name `first`
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  physics_config : dict[str, float]
      Struct containing physical constants for the sphere on which simulation is performed.

  Notes
  -----
  These solve the equations
  ∂𝐮/∂t = −(ζ + f_α)𝐤×𝐮 − ∇(g(h+h_s) + (𝐮⋅𝐮)/2)
  ∂h/∂t = -∇⋅(h𝐮),
  where the planet's axis of rotation can be changed by setting 
  f_α = 2Ω (−cos(λ)cos(ϕ)sin(α) + sin(ϕ)cos(alpha))

  Returns
  -------
  right_hand_side : `ShallowWaterModelState`
      The explicit time tendency, not scaled by ∆t.

  """
  coriolis = (-jnp.cos(grid["physical_coords"][:, :, :, 1]) *
              jnp.cos(grid["physical_coords"][:, :, :, 0]) *
              jnp.sin(physics_config["alpha"]) +
              jnp.sin(grid["physical_coords"][:, :, :, 0]) *
              jnp.cos(physics_config["alpha"]))
  abs_vort = manifold_vorticity(state_in["u"], grid, a=physics_config["radius_earth"]) + 2 * physics_config["angular_freq_earth"] * coriolis
  energy = 0.5 * (state_in["u"][:, :, :, 0]**2 +
                  state_in["u"][:, :, :, 1]**2) + physics_config["gravity"] * (state_in["h"] + state_in["hs"])
  energy_grad = manifold_gradient(energy, grid, a=physics_config["radius_earth"])
  u_tend = abs_vort * state_in["u"][:, :, :, 1] - energy_grad[:, :, :, 0]
  v_tend = -abs_vort * state_in["u"][:, :, :, 0] - energy_grad[:, :, :, 1]
  h_tend = -manifold_divergence(state_in["h"][:, :, :, np.newaxis] * state_in["u"], grid, a=physics_config["radius_earth"])
  return create_state_struct(jnp.stack((u_tend, v_tend), axis=-1), h_tend, state_in["hs"])
