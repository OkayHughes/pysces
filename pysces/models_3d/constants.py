from ..config import device_wrapper


def init_config(Rgas=287.0,
                radius_earth=-1,
                period_earth=7.292e-5,
                gravity=9.81,
                p0=1e5,
                cp=1005.0,
                Rvap=461.50,
                ne=30,
                nu_base=1e15,
                nu_top=2.5e5,
                nu_phi=-1.0,
                nu_dpi=-1.0,
                nu_div_factor=2.5,
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
  nu = nu_base * ((30 / ne) * (radius_earth / radius_earth_base))**3.2
  nu_phi = nu if nu_phi < 0 else nu_phi
  nu_dpi = nu if nu_dpi < 0 else nu_dpi
  return {"Rgas": device_wrapper(Rgas),
          "Rvap": device_wrapper(Rvap),
          "cp": device_wrapper(cp),
          "gravity": device_wrapper(gravity),
          "radius_earth": device_wrapper(radius_earth),
          "period_earth": device_wrapper(period_earth),
          "p0": device_wrapper(p0),
          "diffusion": {"nu": device_wrapper(nu),
                        "nu_phi": device_wrapper(nu_phi),
                        "nu_dpi": device_wrapper(nu_dpi),
                        "nu_div_factor": device_wrapper(nu_div_factor),
                        "nu_top": device_wrapper(nu_top)},
          "reference_profiles": {"T_ref": device_wrapper(T_ref),
                                 "T_ref_lapse": device_wrapper(T_ref_lapse)}}
