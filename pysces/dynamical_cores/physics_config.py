from ..config import device_wrapper
from ..model_info import models, moist_mixing_ratio_models, dry_mixing_ratio_models, cam_se_models, homme_models, variable_kappa_models

boltzmann = 1.38065e-23
avogadro = 6.02214e26

universal_R = boltzmann * avogadro

molec_weight_dry_air = 28.966
molec_weight_water_vapor = 18.016

degrees_of_freedom_igl = {1: 3,
                          2: 5,
                          3: 6}

# Not physically realistic, but achieves close Rgas, cp equivalence
typical_mass_ratios = {frozenset(["N2", "O2"]): {"O2": 0.26,
                                                 "N2": 0.74}}

gas_properties = {"N2": {"num_atoms": 2,
                         "molecular_weight": 28},
                  "O2": {"num_atoms": 2,
                         "molecular_weight": 32},
                  "Ar": {"num_atoms": 1,
                         "molecular_weight": 40}}

def cp_base(dof):
   return universal_R * (1.0 + degrees_of_freedom_igl[dof]/2.0)


def init_physics_config(model,
                        Rgas=universal_R/molec_weight_dry_air,
                        radius_earth=6371e3,
                        period_earth=7.292e-5,
                        gravity=9.81,
                        p0=1e5,
                        cp=1.00464e3,
                        cp_water_vapor=1.810e3,
                        R_water_vapor=universal_R/molec_weight_water_vapor ,
                        epsilon=molec_weight_water_vapor/molec_weight_dry_air,
                        dry_air_species=["N2", "O2"]):
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
    if model in variable_kappa_models:
      physics_config["dry_air_species_Rgas"] = {}
      physics_config["dry_air_species_cp"] = {}
      for species in dry_air_species:
         physics_config["dry_air_species_Rgas"][species] = device_wrapper(universal_R / gas_properties[species]["molecular_weight"])
         physics_config["dry_air_species_cp"][species] = device_wrapper(cp_base(gas_properties[species]["num_atoms"]) / gas_properties[species]["molecular_weight"])
    else:
      physics_config["dry_air_species_Rgas"] = {"dry_air": device_wrapper(Rgas)}
      physics_config["dry_air_species_cp"] = {"dry_air": device_wrapper(cp)}

  physics_config["moisture_species_Rgas"] = {"water_vapor": device_wrapper(R_water_vapor)}
  physics_config["moisture_species_cp"] = {"water_vapor": device_wrapper(cp_water_vapor)}

  return physics_config
  
