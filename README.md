# Endoscape
A series of integrated biophysical calculations for predicting bird and small mammal thermoregulation

The Endoscape simulations require two key components to generatate estimates of thermoregulatory costs for endotherms: (1) a csv file with functional traits of the animal of interest and (2) a directory containing the microhabitat data for making the simulations. I have provided an example of each of these input files to begin generating data with the script.

## Functional traits
The functional traits csv contains  important functional traits that influence heat budgets of birds and mammals and are required for running the Endoscape simulation. These include:

- activity pattern = the primary times of day the animal is active (diurnal or nocturnal)
- mass = mass of the animal (grams)
- insulation_conductivity = thermal conductivity of the insulation (W m-1 K-1)
- density_of_fibers = the density of insulation fibers (fibers cm^-2)
- shortwave_absorptance_dorsal = the absorptance of the dorsal side of the animal to shortwave radiation (expressed as proportion of total)
- shortwave_absorptance_ventral = the absorptance of the ventral side of the animal to shortwave radiation (expressed as proportion of total)
- longwave_absorptance = the absorptance of the animal to longwave radation on (expressed as proportion of total)
- insulation_length_dorsal = the length of the dorsal insulation from the tip of the hair or feather to the base near the skin (m)
- insulation_length_ventral = the length of the ventral insulation from the tip of the hair or feather to the base near the skin (m)
- insulation_depth_dorsal = the depth of the dorsal insulation from the outer surface of the insulation to the skin (m)
- insulation_depth_ventral = the depth of the ventral insulation from the outer surface of the insulation to the skin (m)
- length = length of the animal, not including tail or tail feathers (m)
- width = width of the animal (m)
- height = height of the animal (m)
- emissivity = the emissivity of the animal
- physiology_known = runs an extra set of simulations that incorporates empirical relationships related to water loss, metabolism, and body temperature (0 = unknown; 1 = known); recommended to set to 0.
- version = used for downstream data organization if simulations were run with known physiology; recommended to set to 'no_physiology'
- water_threshold = a user defined value for cooling costs that can determine when an organism seeks shade and ceases activity
- physiology_known_master - = runs an extra set of simulations that incorporates empirical relationships related to water loss, metabolism, and body temperature (0 = unknown; 1 = known); recommended to set to 0. Keep consistent with physiology_known.
- body_temperature_min = use the normothermic body temperature of the animal (C)
- body_temperature_max  = the maximum body temperature of the animal (C)
- lower_critical_temperature = the lower critical temperature of the animal, default set to 25C
- upper_critical_temperature = the upper critical temperature of the animal, default set to 35C
- water_heat_ratio_slope = (recommended to ignore) the slope of the relationship between the ratio of evaporative heat dissipation and metabolic heat production regressed against temperature 
- water_heat_ratio_intercept = (recommended to ignore) the intercept of the relationship between the ratio of evaporative heat dissipation and metabolic heat production regressed against temperature
- body_temperature_slope1 = (recommended to ignore) the slope of the relationship between body temperature regressed against air temperature; first regression if using broken stick
- body_temperature_slope2 = (recommended to ignore) the slope of the relationship between body temperature regressed against air temperature; second regression if using broken stick
- body_temperature_intercept = (recommended to ignore) the intercept of the relationship between body temperature regressed against air temperature
- water_loss_slope = (recommended to ignore) the slope of the relationship between evaporative heat loss regressed against air temperature
- water_loss_intercept = (recommended to ignore) the intercept of the relationship between evaporative heat loss regressed against air temperature

