from ..config import device_wrapper
from .model_info import models

moisture_species_info = {"water_vapor": {"cp": 0.0,
                                         "Rgas": 0.0}}

dry_air_species_info = {"N2": {"Rgas": 0.0,
                               "cp": 0.0},
                        "O2": {"Rgas": 0.0,
                               "cp": 0.0},
                        "CO2": {"Rgas": 0.0,
                                "Cp": 0.0}}


def init_physics_config(model,
                        Rgas=287.0,
                        radius_earth=6371e3,
                        period_earth=7.292e-5,
                        gravity=9.81,
                        p0=1e5,
                        cp=1005.0,
                        Rvap=461.50,
                        epsilon=18.01/28.966):
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

  physics_config = {"gravity": device_wrapper(gravity),
                    "radius_earth": device_wrapper(radius_earth),
                    "period_earth": device_wrapper(period_earth),
                    "p0": device_wrapper(p0),
                    "epsilon": epsilon}
  if model == models.cam_se_upper_atmosphere:
    # todo: find good defaults from CAM infrastructure
    pass
  else:
    physics_config["Rgas"] = device_wrapper(Rgas)
    physics_config["cp"] = device_wrapper(cp)
    physics_config["Rvap"] = device_wrapper(Rvap)
  return physics_config
