from ..config import device_wrapper


def get_physics_config_sw(radius_earth=6371e3, earth_period=7.292e-5, gravity=9.81, alpha=0.0, ne=30):
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
  return {"radius_earth": device_wrapper(radius_earth),
          "earth_period": device_wrapper(earth_period),
          "gravity": device_wrapper(gravity),
          "alpha": device_wrapper(alpha)}
