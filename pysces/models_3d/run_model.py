from ..config import jnp, is_main_proc
from .hyperviscosity import get_ref_states
from .time_stepping import advance_euler, advance_euler_hypervis, ullrich_5stage, advance_euler_sponge
from .model_state import remap_dynamics
from ..distributed_memory.global_communication import global_sum
from ..time_step import time_step_options
from .model_state import advance_dynamics, advance_simple_tracers, wrap_model_state
from ..physics_dynamics_coupling import coupling_types
from .model_info import thermodynamic_variable_names, hydrostatic_models, cam_se_models
from sys import stdout




def check_dynamics_nan(dynamics, model):
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
  is_nan = False
  fields = ["u", thermodynamic_variable_names[model], "d_mass"]
  if model not in hydrostatic_models:
    fields += ["w_i", "phi_i"]
  for field in fields:
    is_nan = is_nan or jnp.any(jnp.isnan(dynamics[field]))
  is_nan = int(is_nan)
  return global_sum(is_nan) > 0

def check_tracers_nan(tracers, model):
  is_nan = False
  for field_name in tracers["moisture_species"].keys():
    is_nan = is_nan or jnp.any(jnp.isnan(tracers["moisture_species"][field_name]))
  for field_name in tracers["tracers"].keys():
    is_nan = is_nan or jnp.any(jnp.isnan(tracers["tracers"][field_name]))
  if model in cam_se_models:
    for field_name in tracers["dry_species"].keys():
      is_nan = is_nan or jnp.any(jnp.isnan(tracers["dry_species"][field_name]))
  is_nan = int(is_nan)
  return global_sum(is_nan) > 0



def advance_coupling_step(state_in, h_grid, v_grid, physics_config, diffusion_config, timestep_config, dims, model, physics_forcing=None):
  physics_dynamics_coupling = timestep_config["physics_dynamics_coupling"]

  dynamics_state = state_in["dynamics"]
  tracer_state = state_in["tracers"]

  if (physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics or 
      physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics):
    tracer_state = advance_simple_tracers([tracer_state, physics_forcing["tracers"]], [1.0, timestep_config["physics_dt"]], model)
  
  if physics_dynamics_coupling == coupling_types.lump_all:
    dynamics_state = advance_dynamics([dynamics_state, physics_forcing["dynamics"]], [1.0, timestep_config["physics_dt"]], model)
    tracer_state = advance_simple_tracers([tracer_state, physics_forcing["tracers"]], [1.0, timestep_config["physics_dt"]], model)


  for q_split in range(timestep_config["tracer_subcycle"]):
    dynamics_state = remap_dynamics(dynamics_state,
                                    v_grid,
                                    physics_config,
                                    len(v_grid["hybrid_b_m"]),
                                    model)
    if (physics_dynamics_coupling == coupling_types.dribble_all or 
        physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics):
      dynamics_state = advance_dynamics([dynamics_state, physics_forcing["dynamics"]], [1.0, timestep_config["tracer_advection"]["dt"]], model)
    if physics_dynamics_coupling == coupling_types.dribble_all:
      tracer_state = advance_simple_tracers([tracer_state, physics_forcing["tracers"]], [1.0, timestep_config["physics_dt"]], model)

    for n_split in range(timestep_config["dynamics_subcycle"]):
      if timestep_config["dynamics"]["step_type"] == time_step_options.Euler:
        dynamics_next = advance_euler(dynamics_state, h_grid, v_grid, physics_config, timestep_config, dims, model)
      elif timestep_config["dynamics"]["step_type"] == time_step_options.RK3_5STAGE:
        dynamics_next = ullrich_5stage(dynamics_state, h_grid, v_grid, physics_config, timestep_config, dims, model)
      else:
        raise ValueError("Unknown dynamics timestep type")
      if "disable_diffusion" not in diffusion_config.keys():
        if timestep_config["hyperviscosity"]["step_type"] == time_step_options.Euler:
          dynamics_next= advance_euler_hypervis(dynamics_next, state_in["static_forcing"], h_grid, v_grid,
                                                physics_config, diffusion_config, timestep_config, dims, model)
        if "enable_sponge_layer" in diffusion_config.keys():
          dynamics_next = advance_euler_sponge(dynamics_next, h_grid, physics_config, diffusion_config, timestep_config, dims, model)

      assert not check_dynamics_nan(dynamics_next, model)
      assert not check_tracers_nan(tracer_state, model)

      dynamics_state, dynamics_next = dynamics_next, dynamics_state
  dynamics_state = remap_dynamics(dynamics_state,
                                  v_grid,
                                  physics_config,
                                  len(v_grid["hybrid_b_m"]),
                                  model)
  return wrap_model_state(dynamics_state,
                          state_in["static_forcing"],
                          tracer_state)


def validate_custom_configuration(state_in,
                                  h_grid, v_grid,
                                  physics_config,
                                  diffusion_config,
                                  timestep_config,
                                  dims,
                                  model):
  pass


def simulate_model(end_time, state_in,
                   h_grid, v_grid, physics_config,
                   diffusion_config, timestep_config,
                   dims, model):
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
  state_n = state_in
  t = 0.0
  times = jnp.arange(0.0, end_time, timestep_config["tracer_advection"]["dt"])
  k = 0
  for t in times:
    if is_main_proc:
      print(f"{k/len(times-1)*100}%")
      stdout.flush()
    state_n = advance_coupling_step(state_n, h_grid, v_grid, physics_config, diffusion_config, timestep_config, dims, model)
    k += 1
  return state_n
