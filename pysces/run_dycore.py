from .dynamical_cores.time_stepping import (advance_dynamics_euler,
                                            advance_hypervis_euler,
                                            advance_dynamics_ullrich_5stage,
                                            advance_sponge_euler)
from .dynamical_cores.model_state import remap_dynamics
from .time_step import time_step_options
from .dynamical_cores.model_state import (sum_dynamics_series,
                                          advance_tracers,
                                          wrap_model_state,
                                          check_dynamics_nan,
                                          check_tracers_nan)
from .dynamical_cores.physics_dynamics_coupling import coupling_types
from .model_info import cam_se_models


def advance_coupling_step(state_in,
                          h_grid,
                          v_grid,
                          physics_config,
                          diffusion_config,
                          timestep_config,
                          dims,
                          model,
                          physics_forcing=None):
  physics_dynamics_coupling = timestep_config["physics_dynamics_coupling"]

  dynamics_state = state_in["dynamics"]
  tracer_state = state_in["tracers"]
  static_forcing = state_in["static_forcing"]
  dribble_dynamics = (physics_dynamics_coupling == coupling_types.dribble_all or
                      physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics)

  if (physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics):
    tracer_state = advance_tracers([tracer_state, physics_forcing["tracers"]],
                                   [1.0, timestep_config["physics_dt"]],
                                   model)

  if physics_dynamics_coupling == coupling_types.lump_all:
    dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                         [1.0, timestep_config["physics_dt"]],
                                         model)
    tracer_state = advance_tracers([tracer_state, physics_forcing["tracers"]],
                                   [1.0, timestep_config["physics_dt"]],
                                   model)

  for q_split in range(timestep_config["tracer_subcycle"]):
    dynamics_state = remap_dynamics(dynamics_state,
                                    state_in["static_forcing"],
                                    v_grid,
                                    physics_config,
                                    len(v_grid["hybrid_b_m"]),
                                    model)
    if dribble_dynamics:
      dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                           [1.0, timestep_config["tracer_advection"]["dt"]],
                                           model)
    if physics_dynamics_coupling == coupling_types.dribble_all:
      tracer_state = advance_tracers([tracer_state, physics_forcing["tracers"]],
                                     [1.0, timestep_config["physics_dt"]],
                                     model)

    for n_split in range(timestep_config["dynamics_subcycle"]):
      if model in cam_se_models:
        moisture_species = tracer_state["moisture_species"]
        dry_air_species = tracer_state["dry_air_species"]
      else:
        moisture_species = None
        dry_air_species = None
      if timestep_config["dynamics"]["step_type"] == time_step_options.Euler:
        dynamics_next = advance_dynamics_euler(dynamics_state,
                                               static_forcing,
                                               h_grid,
                                               v_grid,
                                               physics_config,
                                               timestep_config,
                                               dims,
                                               model,
                                               moisture_species=moisture_species,
                                               dry_air_species=dry_air_species)
      elif timestep_config["dynamics"]["step_type"] == time_step_options.RK3_5STAGE:
        dynamics_next = advance_dynamics_ullrich_5stage(dynamics_state,
                                                        static_forcing,
                                                        h_grid,
                                                        v_grid,
                                                        physics_config,
                                                        timestep_config,
                                                        dims,
                                                        model,
                                                        moisture_species=moisture_species,
                                                        dry_air_species=dry_air_species)
      else:
        raise ValueError("Unknown dynamics timestep type")
      if "disable_diffusion" not in diffusion_config.keys():
        if timestep_config["hyperviscosity"]["step_type"] == time_step_options.Euler:
          dynamics_next = advance_hypervis_euler(dynamics_next,
                                                 static_forcing,
                                                 h_grid,
                                                 v_grid,
                                                 physics_config,
                                                 diffusion_config,
                                                 timestep_config,
                                                 dims,
                                                 model)
        if "enable_sponge_layer" in diffusion_config.keys():
          dynamics_next = advance_sponge_euler(dynamics_next,
                                               h_grid,
                                               physics_config,
                                               diffusion_config,
                                               timestep_config,
                                               dims,
                                               model)

      assert not check_dynamics_nan(dynamics_next, model)
      assert not check_tracers_nan(tracer_state, model)

      dynamics_state, dynamics_next = dynamics_next, dynamics_state
  dynamics_state = remap_dynamics(dynamics_state,
                                  static_forcing,
                                  v_grid,
                                  physics_config,
                                  len(v_grid["hybrid_b_m"]),
                                  model)
  return wrap_model_state(dynamics_state,
                          static_forcing,
                          tracer_state)


def validate_custom_configuration(state_in,
                                  h_grid, v_grid,
                                  physics_config,
                                  diffusion_config,
                                  timestep_config,
                                  dims,
                                  model):
  pass


def init_simulator(h_grid,
                   v_grid,
                   physics_config,
                   diffusion_config,
                   timestep_config,
                   dims,
                   model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  def simulator(state_in, physics_forcing=None):
    state_n = state_in
    t = 0.0
    while True:
      state_n = advance_coupling_step(state_n,
                                      h_grid,
                                      v_grid,
                                      physics_config,
                                      diffusion_config,
                                      timestep_config,
                                      dims,
                                      model,
                                      physics_forcing=physics_forcing)
      t += timestep_config["physics_dt"]
      physics_forcing = yield t, state_n
  return simulator
