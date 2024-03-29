# Endoscape
A series of integrated biophysical calculations for predicting bird and small mammal thermoregulatory costs.

The Endoscape simulations require three key components to generatate estimates of thermoregulatory costs for endotherms: (1) a csv file with functional traits of the animal of interest, (2) a csv file containing information about the sites, and (3) a directory containing the microhabitat data for making the simulations. I have provided an example of each of these input files to begin generating data with the script.

The functional trait data provided allow users to generate hourly thermoregulatory costs for an White-tailed antelope squirrel (Ammospermophilus leucurus).

## Functional traits
The functional traits csv contains important functional traits that influence heat budgets of birds and mammals and are required for running the Endoscape simulation. These include:

- species = species name (prefer latin, no spaces)
- common_name = species common name (no spaces)
- type = mammal or bird
- activity pattern = activity pattern = the general time of day when the animal is active (diurnal or nocturnal)
- mass = mass of the animal (grams)
- insulation_conductivity = thermal conductivity of the insulation (W m^-1 K^-1)
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
- body_temperature_min = use the normothermic body temperature of the animal (C)
- body_temperature_max  = the maximum body temperature of the animal (C)
- lower_critical_temperature = the lower critical temperature of the animal, default set to 25C
- upper_critical_temperature = the upper critical temperature of the animal, default set to 35C
- water_heat_ratio_slope = (recommended to ignore, leave empty) the slope of the relationship between the ratio of evaporative heat dissipation and metabolic heat production regressed against temperature 
- water_heat_ratio_intercept = (recommended to ignore, leave empty) the intercept of the relationship between the ratio of evaporative heat dissipation and metabolic heat production regressed against temperature
- body_temperature_slope1 = (recommended to ignore, leave empty) the slope of the relationship between body temperature regressed against air temperature; first regression if using broken stick
- body_temperature_slope2 = (recommended to ignore, leave empty) the slope of the relationship between body temperature regressed against air temperature; second regression if using broken stick
- body_temperature_intercept = (recommended to ignore, leave empty) the intercept of the relationship between body temperature regressed against air temperature
- water_loss_slope = (recommended to ignore, leave empty) the slope of the relationship between evaporative heat loss regressed against air temperature
- water_loss_intercept = (recommended to ignore, leave empty) the intercept of the relationship between evaporative heat loss regressed against air temperature

## Site characteristics

A csv file with site characteristics.

- site_id = a unique, arbitrary numerical ID for each site
- site = name of site (not required)
- longitude = longitude in decimal degrees
- latitude = latitude in decimal degrees
- elevation = elevation (m)
- soil = maxiumum depth of soil at site (m)

## Microhabitat conditions

Estimates of microclimate conditions for every hour of the day for an average day of every month at a single site created using NicheMapR.

- DOY	= day of year
- TIME	= time (min)
- D0cm	= soil surface temperature (C)
- D2.5cm	= soil temperature (C) at 2.5 cm
- D5cm	= soil temperature (C) at 5 cm
- D10cm	= soil temperature (C) at 10 cm
- D15cm	= soil temperature (C) at 15 cm
- D20cm	= soil temperature (C) at 20 cm
- D30cm	= soil temperature (C) at 30 cm
- D50cm	= soil temperature (C) at 50 cm
- D100cm	= soil temperature (C) at 100 cm
- D200cm	= soil temperature (C) at 200 cm
- Tair	= air temperature (C) at 2 m from the soil surface (used for birds)
- Tair_mammals = air temperature at 0.1 cm from the soil surface (used for mammals)
