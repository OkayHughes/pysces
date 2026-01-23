from ..config import device_wrapper
from ..model_info import models, moist_mixing_ratio_models, dry_mixing_ratio_models, cam_se_models, homme_models

boltzmann = 1.38065e-23
avogadro = 6.02214e26
universal_R = boltzmann * avogadro
molec_weight_dry_air = 28.966
molec_weight_water_vapor = 18.016



def init_physics_config(model,
                        Rgas=universal_R/molec_weight_dry_air,
                        radius_earth=6371e3,
                        period_earth=7.292e-5,
                        gravity=9.81,
                        p0=1e5,
                        cp=1.00464e3,
                        cp_water_vapor=1.810e3,
                        R_water_vapor=universal_R/molec_weight_water_vapor ,
                        epsilon=molec_weight_water_vapor/molec_weight_dry_air):
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
  physics_config["Rgas"] = device_wrapper(Rgas)
  physics_config["cp"] = device_wrapper(cp)
  if model in cam_se_models:
    physics_config["dry_air_species_Rgas"] = {"dry_air": device_wrapper(Rgas)}
    physics_config["dry_air_species_cp"] = {"dry_air": device_wrapper(cp)}

  physics_config["moisture_species_Rgas"] = {"water_vapor": R_water_vapor}
  physics_config["moisture_species_cp"] = {"water_vapor": cp_water_vapor}

  return physics_config
  
