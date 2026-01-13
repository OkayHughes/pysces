from ..config import device_wrapper, jnp


def init_config(Rgas=287.0,
                radius_earth=-1,
                period_earth=7.292e-5,
                gravity=9.81,
                p0=1e5,
                cp=1005.0,
                Rvap=461.50,
                ne=30,

                T_ref=288.0,
                T_ref_lapse=0.0065):
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
  radius_earth_base = 6371e3
  if radius_earth < 0:
    radius_earth = radius_earth_base
  return {"Rgas": device_wrapper(Rgas),
          "Rvap": device_wrapper(Rvap),
          "cp": device_wrapper(cp),
          "gravity": device_wrapper(gravity),
          "radius_earth": device_wrapper(radius_earth),
          "period_earth": device_wrapper(period_earth),
          "p0": device_wrapper(p0),
          "reference_profiles": {"T_ref": device_wrapper(T_ref),
                                 "T_ref_lapse": device_wrapper(T_ref_lapse)}}
