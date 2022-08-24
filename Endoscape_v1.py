#---------------------#
#------DESCRIPTION------#
#---------------------#
'''The following version of Endoscape predicts thermoregulatory requirements
for small mammals and birds. The script requires a few files to run. The first
file is endotherm_properties.csv, which contains the species-specific traits
required to make the calculations. The simulation also requires a database of sites
and the site-specific dataframes for each site.'''

#---------------------#
#------LIBRARIES------#
#---------------------#

import numpy as numpy
import math as math
import pandas as pandas
import glob as glob
import multiprocessing as multiprocessing
import itertools as itertools
import random as random

#--------------------#
#-----CONSTANTS------#
#--------------------#

SOLAR_CONSTANT = 1360. #W*m^-2
TAU = 0.7 #0.7 clear day
STEFAN_BOLTZMANN = 5.670373*10**(-8) #W*m^-2*K^-4
ALBEDO = 0.4 #ground reflectance (albedo) of dry sandy soil ranges 0.25-0.45
E_G = 0.9 ##surface emissivity of sandy soil w/<2% organic matter approx 0.88
OMEGA = math.pi/12.

#-------------------#
#-----CLASSESS------#
#-------------------#

class Individual():
    def __init__(self,endotherm_type,scenario,common_name,latitude,longitude,elevation,length,width,height,insulation_length_dorsal,insulation_length_ventral,density_of_fibers,insulation_depth_dorsal,insulation_depth_ventral,physiology_known,Tb,Tb_max,LCT,UCT,mass,shortwave_absorbance_dorsal,shortwave_absorbance_ventral,longwave_absorbance,water_heat_ratio_slope,water_heat_ratio_intercept,water_loss_slope,water_loss_intercept,Tb_slope1,Tb_slope2,Tb_intercept,windspeed,posture,fiber_density,shade,feather_depths,orientation,shape,water_threshold,insulation_conductivity,activity_pattern,site_number,modern_soils,historic_soils,soil_depth):
        self.type = endotherm_type
        self.scenario = scenario
        self.common_name = common_name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = elevation
        self.MASS = mass #grams, 22.9 for a white crowned sparrow
        self.dehydration_mass = mass
        self.density_of_fibers = density_of_fibers + (density_of_fibers*fiber_density)#fiber number is density
        self.animal_slope = 0.0 + (60.0 * posture)
        self.orientation = orientation
        self.body_volume = (self.MASS/0.99)/1000000#m^3; from Seamans et al 1995, determination of bird density of 12 species
        self.T_b = Tb
        self.Tes_past = 0.0
        self.Qgen_past = 0.0
        self.Tb_max = Tb_max
        self.Tb_slope1 = Tb_slope1
        self.Tb_slope2 = Tb_slope2
        self.Tb_intercept = Tb_intercept
        self.lower_critical_temperature = LCT
        self.upper_critical_temperature = UCT
        self.thermoneutral_zone_range = self.upper_critical_temperature - self.lower_critical_temperature
        self.insulation_depth_dorsal = insulation_depth_dorsal + (insulation_depth_dorsal * feather_depths) #0.02 * (self.MASS**(1./5.))#Bakken 1976,m
        self.insulation_depth_ventral = insulation_depth_ventral + (insulation_depth_ventral * feather_depths)#0.02 * (self.MASS**(1./5.))#Bakken 1976,m
        self.A_radius = length/2.0 - ((length/2.0) * (shape*0.35)) #radius of length of prolate spheroid
        self.B_radius = width/2.0 - ((width/2.0) * (shape*0.35))#radius of width of prolate spheroid
        self.C_radius = height/2.0 - ((height/2.0) * (shape*0.35))#radius of height of prolate spheroid
        self.insulation_length_dorsal = insulation_length_dorsal
        self.insulation_length_ventral = insulation_length_ventral
        self.D = (self.body_volume)**(1.0/3.0) #characteristic dimension
        self.H = self.A_radius * 2.0 #length of cylinder, meters, only used if you want to use the rounded cyclinder for view factor
        self.S = 1.0 + shade #self.grid[self.x,self.y] #proportion of animal exposed to direct solar radiation (non-shaded)
        self.A_S_dorsal = shortwave_absorbance_dorsal #absorptance of organism to shortwave radiation; solar absorptivity is approx 0.9 for lizards, 0.75 for most birds and mammals [Gates 1980, Buckley 2008]
        self.A_S_ventral = shortwave_absorbance_ventral
        self.A_L = longwave_absorbance #absorptance of organism to longwave radiation; thermal absorptivity approx 0.965 [Bartlett and Gates 1967, Buckley 2008]
        self.E_S = 0.97 #emissivity of organism
        self.conductivity_insulation = insulation_conductivity #thermal conductivity of insulation W/m C - assumed 0.027 in Porter and Kearney 2010, PNAS or 0.0257 for still air, or 0.075 from Wolf and Walsberg 2000,0.11 was from Webster
        self.A_radius_insulation = self.A_radius
        self.B_radius_insulation = self.B_radius
        self.C_radius_insulation = self.C_radius
        self.surface_area_outer = (4*math.pi*(((((self.A_radius_insulation*self.B_radius_insulation)**1.6)+((self.A_radius_insulation*self.C_radius_insulation)**1.6)+((self.B_radius_insulation*self.C_radius_insulation)**1.6))/3.0)**(1/1.6)))#*1.3
        self.surface_area_inner = 1.23 * self.surface_area_outer
        self.surface_area_outer_cold = self.surface_area_outer * 0.73
        self.surface_area_inner_cold = self.surface_area_inner * 0.73
        self.surface_area_outer_splayed  = self.surface_area_outer * 1.2
        self.surface_area_inner_splayed  = self.surface_area_inner * 1.2
        self.conductance_insulation_dorsal = self.conductivity_insulation/self.insulation_depth_dorsal#W K-1
        self.conductance_insulation_ventral = self.conductivity_insulation/self.insulation_depth_ventral#W K-1
        self.conductivity_skin = 2.8 #thermal conductivity of skin W m-1 C-1,vasoconstricted
        self.skin_depth = 0.01 * (self.D**(0.6))#Bakken 1976, m
        self.conductance_skin = self.conductivity_skin/self.skin_depth#Bakken 1976,W m-2 K-1
        self.conductance_skin_insulation_dorsal = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))
        self.conductance_skin_insulation_ventral = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))
        self.windspeed = windspeed #m/s
        self.physiology_known = physiology_known# 1 is known, 0 is unknown
        self.water_heat_ratio_slope = water_heat_ratio_slope #slope of relationship between EHL and MHP from empirical data
        self.water_heat_ratio_intercept = water_heat_ratio_intercept #intercept of relationship between EHL and MHP from empirical data
        self.water_heat_ratio = self.water_heat_ratio_intercept**(self.water_heat_ratio_slope*self.T_b)#amount of water lost relative to heat produced
        self.resting_metabolic_rate = 0.0#Watts
        self.water_loss_slope = water_loss_slope#slope of relationship between EHL and air temperature from empirical data
        self.water_loss_intercept = water_loss_intercept#intercept of relationship between EHL and air temperature from empirical data
        self.evaporative_heat_loss = 0.0 #Watts
        self.water_heat_balance = 0.0 #Watts
        self.energy_balance = 0.0 #Watts
        self.water_loss_mass = 0.0 #grams
        self.excess_proportion_of_mass_lost = 0.0 #grams
        self.excess_water = 0.0 # grams
        self.metabolic_heat_required = 0.0 #Watts
        self.water_loss_required = 0.0 #grams
        self.proportion_water_loss_required = 0.0 #proportion of mass
        self.excess_metabolic_heat = 0.0 #Watts
        self.Qgen_water = 0.0 #Watts
        self.Qgen_heat = 0.0#Watts
        self.activity_above_thermal_stress = 0.0
        self.activity_above_water_stress = 0.0
        self.heat_stress = 0.0
        self.water_stress = 0.0
        self.Qgen_water_sum = 0.0
        self.Qgen_water_threshold = water_threshold
        self.active_hours = 0.0
        self.activity_pattern = activity_pattern
        self.site = site_number
        self.modern_soils = modern_soils
        self.historic_soils = historic_soils
        self.daylight = 0.0
        self.Tref = 0.0
        self.Te_dorsal = 0.0
        self.Ke_d = 0.0
        self.Hi_d = 0.0
        self.Ri_d = 0.0
        self.soil_ref = 'above'
        self.soil_depth = soil_depth
    
    def orbit_correction(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        return 1 + 2 * 0.01675 * math.cos((((2*math.pi)/365))*day)
    
    def direct_solar_radiation(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        return self.orbit_correction(day)*SOLAR_CONSTANT
        
    def f(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        return 279.575 + (0.9856 * day)
    
    def ET(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        _f = math.radians(self.f(day))
        return (-104.7*math.sin(_f)+596.2*math.sin(2*_f)+4.3*math.sin(3*_f)-12.7*math.sin(4*_f)-429.3*math.cos(_f)-2.0*math.cos(2*_f)+19.3*math.cos(3*_f))/3600.
    
    def LC(self,lon):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        return ((lon%15)*4.0)/60

    def t0(self,lc,et):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        t = 12 + lc - et
        return t

    def hour(self,t,t_zero):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        h = 15*(t-t_zero)
        return h

    def declin(self,day):
        'a function that calculates the declination angle'
        return math.degrees(math.asin(0.39785* math.sin(math.radians(278.97 + 0.9856 * day + 1.9165 * math.sin(math.radians(356.6 + 0.9856 * day))))))
        
    def zenith(self,day,t):
        'a function that calculates the zenith angle'
        if math.acos(math.sin(math.radians(self.latitude))*math.sin(math.radians(self.declin(day))) + math.cos(math.radians(self.latitude))*math.cos(math.radians(self.declin(day)))*math.cos(math.radians(self.hour(t,(self.t0(self.LC(self.longitude),self.ET(day))))))) >= 0.:
            return math.acos(math.sin(math.radians(self.latitude))*math.sin(math.radians(self.declin(day))) + math.cos(math.radians(self.latitude))*math.cos(math.radians(self.declin(day)))*math.cos(math.radians(self.hour(t,(self.t0(self.LC(self.longitude),self.ET(day)))))))
        else:
            return 0.
            
    def azimuth(self,day,t):
        'a function that calculates the azimuth angle'
        return (math.acos(-1.*(-(math.sin(math.radians(self.declin(day)))-(math.cos(self.zenith(day,t))*math.sin(math.radians(self.latitude)))))/((math.cos(math.radians(self.latitude)))*math.sin(self.zenith(day,t)))))
        
    def animal_angle(self,day,t):
        'a function that calculates the angle of the animal related to the sun'
        return (math.acos((math.cos(math.radians(self.animal_slope)) * math.cos(self.zenith(day,t))) + (math.sin(math.radians(self.animal_slope)) * math.sin(self.zenith(day,t)) * math.cos(self.azimuth(day,t)-math.radians(180.- self.animal_slope)))))
            
    def m(self,day,hrs):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        p_a = 101.3*math.exp(-self.altitude/8200)
        if math.cos(self.zenith(day,hrs))>=0.:
            return p_a/(101.3*(math.cos(self.zenith(day,hrs))))
        else:
            return 0.
            
    def hS0(self,day,hrs):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        z = self.zenith(day,hrs)
        if math.cos(z)>= 0.:
            return self.direct_solar_radiation(day)*(math.cos(z))
        else:
            return 0.
            
    def hS(self,day, hrs, tau):
        'a function required to estimate solar radiation from Campbell and Norman (2010)'
        self.daylight = self.hS0(day,hrs)*tau**self.m(day,hrs)
        return self.hS0(day,hrs)*tau**self.m(day,hrs)

    def diffuse_solar(self,day,hrs,tau):
        'a function for diffuse solar radiation'
        return self.hS0(day,hrs)*0.3*(1.-(TAU**self.m(day,hrs)))

    def reflected_radiation(self,day,t,tau):
        'a function for reflected solar radiation'
        return ALBEDO*self.hS(day,t,tau)

    def view_factor_prolate_spheroid(self,animal_angle,a,b):
        'a function for view factor for a proloate spheroid'
        view_factor = ((math.sqrt(1+((((b/a)**2)-1)*((math.cos(math.radians(90.-math.degrees(animal_angle))))**2))))/((2*(b/a))+((2*(math.asin(math.sqrt((1-((b/a)**2))))))/(math.sqrt((1-((b/a)**2)))))))
        return view_factor + (view_factor * self.orientation)
    
    def air_temp(self,mins,maxes,day,hour):
        'a function grabs air temperature data for birds from NicheMapR output files'
        if self.scenario =='historic':
            self.Tref = self.historic_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
        elif self.scenario == 'modern':
            self.Tref = self.modern_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
        return(self.Tref)

    def ground_temp(self,day,hour):
        'a function grabs data from NicheMapR and is used exclusively for longwave radiation from the ground for mammals'
        self.Tref = 0.0
        if self.type == 'mammal':
            if self.soil_ref == 'above':
                    if self.scenario == 'historic':
                        self.Tref = self.historic_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                    elif self.scenario == 'modern':
                        self.Tref = self.modern_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
            else:
                    if self.scenario == 'historic':
                        if self.soil_depth < 0.05:
                            self.Tref = self.historic_soils['D2.5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.05 and self.soil_depth < 0.1:
                            self.Tref = self.historic_soils['D5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.1 and self.soil_depth < 0.15:
                            self.Tref = self.historic_soils['D10cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.15 and self.soil_depth < 0.2:
                            self.Tref = self.historic_soils['D15cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.2 and self.soil_depth < 0.3:
                            self.Tref = self.historic_soils['D20cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.3 and self.soil_depth < 0.5:
                            self.Tref = self.historic_soils['D30cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.5 and self.soil_depth < 1.0:
                            self.Tref = self.historic_soils['D50cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 1.0:
                            self.Tref = self.historic_soils['D100cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                    elif self.scenario == 'modern':
                        if self.soil_depth < 0.05:
                            self.Tref = self.modern_soils['D2.5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.05 and self.soil_depth < 0.1:
                            self.Tref = self.modern_soils['D5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.1 and self.soil_depth < 0.15:
                            self.Tref = self.modern_soils['D10cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.15 and self.soil_depth < 0.2:
                            self.Tref = self.modern_soils['D15cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.2 and self.soil_depth < 0.3:
                            self.Tref = self.modern_soils['D20cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.3 and self.soil_depth < 0.5:
                            self.Tref = self.modern_soils['D30cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.5 and self.soil_depth < 1.0:
                            self.Tref = self.modern_soils['D50cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 1.0:
                            self.Tref = self.modern_soils['D100cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
        return self.Tref
                
    def ground_temp_NicheMapR(self,day,hour):
        'a function that grabs temperatures from NicheMapR output files'
        self.Tref = 0.0
        if self.type == 'mammal':
            if self.soil_ref == 'above':
                    if self.scenario == 'historic':
                        self.Tref = self.historic_soils.Tair_mammals.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                    elif self.scenario == 'modern':
                        self.Tref = self.modern_soils.Tair_mammals.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
            else:
                    if self.scenario == 'historic':
                        if self.soil_depth < 0.05:
                            self.Tref = self.historic_soils['D2.5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.05 and self.soil_depth < 0.1:
                            self.Tref = self.historic_soils['D5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.1 and self.soil_depth < 0.15:
                            self.Tref = self.historic_soils['D10cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.15 and self.soil_depth < 0.2:
                            self.Tref = self.historic_soils['D15cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.2 and self.soil_depth < 0.3:
                            self.Tref = self.historic_soils['D20cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.3 and self.soil_depth < 0.5:
                            self.Tref = self.historic_soils['D30cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.5 and self.soil_depth < 1.0:
                            self.Tref = self.historic_soils['D50cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 1.0:
                            self.Tref = self.historic_soils['D100cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                    elif self.scenario == 'modern':
                        if self.soil_depth < 0.05:
                            self.Tref = self.modern_soils['D2.5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.05 and self.soil_depth < 0.1:
                            self.Tref = self.modern_soils['D5cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.1 and self.soil_depth < 0.15:
                            self.Tref = self.modern_soils['D10cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.15 and self.soil_depth < 0.2:
                            self.Tref = self.modern_soils['D15cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.2 and self.soil_depth < 0.3:
                            self.Tref = self.modern_soils['D20cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.3 and self.soil_depth < 0.5:
                            self.Tref = self.modern_soils['D30cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 0.5 and self.soil_depth < 1.0:
                            self.Tref = self.modern_soils['D50cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                        elif self.soil_depth >= 1.0:
                            self.Tref = self.modern_soils['D100cm'].iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
        elif self.type == 'bird':
            if self.scenario == 'historic':
                        self.Tref = self.historic_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
            elif self.scenario == 'modern':
                        self.Tref = self.modern_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
        return self.Tref

    def longwave_sky(self,temperature):
        'a function for long wave radiation from the sky'
        return 53.1*10**-14*(temperature+273.15)**6.

    def longwave_ground(self,temperature):
        'a function for long wave radiation from the ground'
        return E_G*STEFAN_BOLTZMANN*(temperature+273.15)**4.
            
    def radiative_conductance(self,mins,maxes,day,hour,soil_mins,soil_maxes):
        'a function for radiative conductance'
        if self.type == "bird":
            return (4.0 * (self.surface_area_outer/2.0) * STEFAN_BOLTZMANN * self.E_S * ((self.air_temp(mins,maxes,day,hour)+273.15)**3.)) #Bakken, for bird, divided by half for 2D heat flow
        elif self.type == "mammal":
            return (4.0 * (self.surface_area_outer/2.0) * STEFAN_BOLTZMANN * self.E_S * ((self.ground_temp_NicheMapR(day,hour)+273.15)**3.)) #Bakken, for mammal, divided by half for 2D heat flow, W K-1
        else:
            print("Warning: incorrect endotherm type")
    
    def convective_conductance(self,mins,maxes,day,hour,soil_mins,soil_maxes,windspeed):
        'a function for convective conductance'
        if self.type == "bird":
            air_pressure = (101325.*(1.-(2.2569*10**-5)*self.altitude)**5.2553)
            temp_K = self.air_temp(mins,maxes,day,hour) + 273.15
            thermal_conductivity = (2.4525*10**-2)+((7.038*10**-5)*(temp_K-273.15))
            air_density = air_pressure/(287.04*temp_K)
            dynamic_viscosity = (1.8325*10**-5)*((296.16+120.)/(temp_K+120.))*((temp_K/296.16)**1.5)
            Reynolds = (air_density*windspeed*self.D)/dynamic_viscosity
            Nusselt = 0.37*(Reynolds**0.6)#this value was based off of Gates 1980,equation 9.64, assumes a sphere shape
            hc = (Nusselt*thermal_conductivity)/self.D
            hc_enhanced = hc * 1.3 #if bird, enhancement based upon Mitchell 1976
            return hc_enhanced * (self.surface_area_outer/2.0)
        elif self.type == "mammal":
            air_pressure = (101325.*(1.-(2.2569*10**-5)*self.altitude)**5.2553)
            temp_K = self.ground_temp_NicheMapR(day,hour) + 273.15
            thermal_conductivity = (2.4525*10**-2)+((7.038*10**-5)*(temp_K-273.15))
            air_density = air_pressure/(287.04*temp_K)
            dynamic_viscosity = (1.8325*10**-5)*((296.16+120.)/(temp_K+120.))*((temp_K/296.16)**1.5)
            Reynolds = (air_density*windspeed*self.D)/dynamic_viscosity
            Nusselt = 0.37*(Reynolds**0.6)#this value was based off of Gates 1980,equation 9.64
            hc = (Nusselt*thermal_conductivity)/self.D
            hc_enhanced = hc * 1.7 #if mammal, enhancement based upon Mitchell 1976
            return hc_enhanced * (self.surface_area_outer/2.0)  #W m-2 K-1
        else:
            print("Warning: incorrect endotherm type in convective conductance")
            
    def probability(self,day,time):
        'a function calculates the propability that a ray of sunlight passes through the insulation to hit the skin directly'
        return (self.density_of_fibers * 10000.) * 0.00003 * ((((1. + (math.tan(numpy.arccos(self.insulation_depth_dorsal/self.insulation_length_dorsal))**2.))*(1. + (math.tan(self.animal_angle(day,time))**2.)))-((1.+ math.tan(numpy.arccos(self.insulation_depth_dorsal/self.insulation_length_dorsal))*math.tan(self.animal_angle(day,time))*math.cos(self.azimuth(day,time)))**2.))**(1./2.))
    
    def radiation_abs(self,area,mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed):
        'a function calculates the total amount of absorbed radiation'
        if area == "dorsal" and self.type == "bird":
                return ((self.S*(self.A_S_dorsal + ((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed)+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))/(self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_dorsal)) * (2.-self.A_S_dorsal)))*((self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS0(day,hour))+((self.diffuse_solar(day,hour,tau)))))+(self.A_L*((self.longwave_sky(self.air_temp(mins,maxes,day,hour)))))
        elif area == "dorsal" and self.type == "mammal":
            if self.activity_pattern == 'diurnal' and self.daylight > 0:
                return ((self.S*(self.A_S_dorsal + ((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed)+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))/(self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_dorsal)) * (2.-self.A_S_dorsal)))*((self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS0(day,hour))+((self.diffuse_solar(day,hour,tau)))))+(self.A_L*((self.longwave_sky(self.ground_temp_NicheMapR(day,hour)))))
            else:
                return (self.S*(self.A_S_dorsal*(self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS0(day,hour)))+((self.diffuse_solar(day,hour,tau))))+(self.A_L*((self.longwave_sky(self.ground_temp_NicheMapR(day,hour)))))
        elif area == "ventral" and self.type == "bird":
                return ((self.S*(self.A_S_ventral + ((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed)+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))/(self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_ventral)) * (2.-self.A_S_ventral)))*((self.reflected_radiation(day,hour,tau))))+(self.A_L*(self.longwave_ground(self.ground_temp_NicheMapR(day,hour))))
        elif area == "ventral" and self.type == "mammal":
            if self.activity_pattern == 'diurnal' and self.daylight > 0:
                return ((self.S*(self.A_S_ventral + ((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed)+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))/(self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_ventral)) * (2.-self.A_S_ventral)))*((self.reflected_radiation(day,hour,tau))))+(self.A_L*(self.longwave_ground(self.ground_temp(day,hour))))
            else:
                return (self.S*(self.A_S_ventral*(self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS0(day,hour))+((self.reflected_radiation(day,hour,tau))))+(self.A_L*((self.longwave_ground(self.ground_temp(day,hour))))))
        else:
            print("Warning in radiation absorbed")
  
    def effective_conductance(self,area,radiative_conductance, convective_conductance):
        'a function for effective conductance'
        if area == "dorsal":
            self.Ke_d = (self.conductance_skin_insulation_dorsal*(radiative_conductance + (convective_conductance)))/(self.conductance_skin_insulation_dorsal + radiative_conductance + convective_conductance)
            return(self.Ke_d)
        elif area == "ventral":
            return (self.conductance_skin_insulation_ventral*(radiative_conductance + (convective_conductance)))/(self.conductance_skin_insulation_ventral + radiative_conductance + convective_conductance)
        else:
            print("Warning in effective conductance")
            
    def Ke_overall(self,mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed):
        'a function for total effective conductance'
        field_dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
        field_ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
        return field_dorsal_conductance + field_ventral_conductance

    def Q_gen(self,mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed):
        'a function calculates dorsal and ventral net sensible heat flux from both sides independently and then adds them together'
        if self.type == "bird":
            dorsal_Te = self.air_temp(mins,maxes,day,hour) + (((self.radiation_abs('dorsal',mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.air_temp(mins,maxes,day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, bird
            ventral_Te = self.air_temp(mins,maxes,day,hour) + (((self.radiation_abs('ventral',mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.air_temp(mins,maxes,day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, bird
            dorsal_conductance_standard = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,self.windspeed))
            ventral_conductance_standard = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,self.windspeed))
            dorsal_conductance_field = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
            ventral_conductance_field = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
            relative_conductance_dorsal = dorsal_conductance_field/dorsal_conductance_standard
            relative_conductance_ventral = ventral_conductance_field/ventral_conductance_standard
            Tes_dorsal = relative_conductance_dorsal*dorsal_Te + (1. - relative_conductance_dorsal)*self.T_b
            Tes_ventral = relative_conductance_ventral*ventral_Te + (1. - relative_conductance_ventral)*self.T_b
            Qgen_dorsal = dorsal_conductance_field*(self.T_b - Tes_dorsal)
            Qgen_ventral = ventral_conductance_field*(self.T_b - Tes_ventral)
        if self.type == "mammal":
            dorsal_Te = self.ground_temp_NicheMapR(day,hour) + (((self.radiation_abs('dorsal',mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.ground_temp_NicheMapR(day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, mammal
            ventral_Te = self.ground_temp_NicheMapR(day,hour) + (((self.radiation_abs('ventral',mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.ground_temp_NicheMapR(day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, mammal
            dorsal_conductance_standard = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,self.windspeed))
            ventral_conductance_standard = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,self.windspeed))
            dorsal_conductance_field = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
            ventral_conductance_field = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
            relative_conductance_dorsal = dorsal_conductance_field/dorsal_conductance_standard
            relative_conductance_ventral = ventral_conductance_field/ventral_conductance_standard
            Tes_dorsal = relative_conductance_dorsal*dorsal_Te + (1. - relative_conductance_dorsal)*self.T_b
            Tes_ventral = relative_conductance_ventral*ventral_Te + (1. - relative_conductance_ventral)*self.T_b
            Qgen_dorsal = dorsal_conductance_field*(self.T_b - Tes_dorsal)
            Qgen_ventral = ventral_conductance_field*(self.T_b - Tes_ventral)
        return Qgen_dorsal + Qgen_ventral
        
    def operative_temperature(self,mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed):
        'a function for operative temperature'
        if self.type == "bird":
            self.dorsal_Te = self.air_temp(mins,maxes,day,hour) + (((self.radiation_abs('dorsal',mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.air_temp(mins,maxes,day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, bird
            ventral_Te = self.air_temp(mins,maxes,day,hour) + (((self.radiation_abs('ventral',mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.air_temp(mins,maxes,day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, bird
            dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))
            ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))
            total_conductance = dorsal_conductance + ventral_conductance
            return ((dorsal_conductance * self.dorsal_Te)+(ventral_conductance * ventral_Te))/total_conductance
        elif self.type == "mammal":
            self.dorsal_Te = self.ground_temp_NicheMapR(day,hour) + (((self.radiation_abs('dorsal',mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.ground_temp_NicheMapR(day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, mammmal
            ventral_Te = self.ground_temp_NicheMapR(day,hour) + (((self.radiation_abs('ventral',mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.ground_temp_NicheMapR(day,hour))**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))+self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes))) #Bakken, mammal
            dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))
            ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,windspeed))
            total_conductance = dorsal_conductance + ventral_conductance
            return ((dorsal_conductance * self.dorsal_Te)+(ventral_conductance * ventral_Te))/total_conductance
        else:
            print("Error in operative temperature")
    
    def standard_operative_temperature(self,mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed):
        'a function for standard operative temperature'
        dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,self.windspeed))
        ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,self.windspeed))
        standard_overall_conductance = dorsal_conductance + ventral_conductance
        field_dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
        field_ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(mins,maxes,day,hour,soil_mins,soil_maxes),self.convective_conductance(mins,maxes,day,hour,soil_mins,soil_maxes,field_windspeed))
        field_total_conductance = field_dorsal_conductance + field_ventral_conductance
        relative_conductance = field_total_conductance/standard_overall_conductance
        return self.T_b - relative_conductance*(self.T_b - self.operative_temperature(mins,maxes,day,hour,tau,soil_mins,soil_maxes,field_windspeed))
        
    def update_skin_conductance(self,Tes):
        'a function that adjusts the skin conductance based on operative temperature'
        if Tes <= self.lower_critical_temperature:
            self.conductivity_skin = 0.204
        elif Tes >= self.upper_critical_temperature:
            self.conductivity_skin = 2.8
        elif Tes > self.lower_critical_temperature and Tes < self.upper_critical_temperature:
           self.conductivity_skin = 2.8 * ((Tes - self.lower_critical_temperature)/self.thermoneutral_zone_range)
        self.conductance_skin = self.conductivity_skin/self.skin_depth
        self.conductance_skin_insulation_dorsal = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))
        self.conductance_skin_insulation_ventral = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))
    
    def update_Tb(self,Tes,Tb_max):
        'a function is only used if physiological sensitivities are known, not used for the publication'
        if Tes <= 30.0:
            self.T_b  = (self.Tb_slope1 * 30.) + (self.Tb_slope2 * (30.0**2)) + self.Tb_intercept
        elif Tes > 30 and Tes <= 55:
            self.T_b  = (self.Tb_slope1 * Tes) + (self.Tb_slope2 * (Tes**2)) + self.Tb_intercept
        else:
            self.T_b  = (self.Tb_slope1 * 55.) + (self.Tb_slope2 * (55.**2)) + self.Tb_intercept
        if self.T_b > Tb_max:
            self.T_b = Tb_max
            
    def update_water_loss_rate(self,Tes):#2.41 J mg^-1;rates are g/hour
        'a function is only used if physiological sensitivities are known, not used for the publication'
        if Tes <= 30.0:
            self.evaporative_heat_loss = (self.water_loss_intercept*math.exp((self.water_loss_slope*30.)))/1000.
        elif Tes > 30:
            self.evaporative_heat_loss = (self.water_loss_intercept*math.exp((self.water_loss_slope*Tes)))/1000.
            
    def update_water_heat_ratio(self,Tes):
        'a function is only used if physiological sensitivities are known, not used for the publication'
        if Tes <= 30.0:
            self.water_heat_ratio = self.water_heat_ratio_intercept*math.exp((self.water_heat_ratio_slope*30.0))
        elif Tes > 30:
            self.water_heat_ratio = self.water_heat_ratio_intercept*math.exp((self.water_heat_ratio_slope*Tes))
        self.resting_metabolic_rate = self.evaporative_heat_loss/self.water_heat_ratio
            
    def update_water_heat_balance(self):
        'a function is only used if physiological sensitivities are known, not used for the publication'
        self.water_heat_balance = self.evaporative_heat_loss - self.resting_metabolic_rate
            
    def update_energy_balance(self,Qgen):
        'a function is only used if physiological sensitivities are known, not used for the publication'
        self.energy_balance = Qgen + self.water_heat_balance
        if self.energy_balance > 0.0 and Qgen < 0.0:#if you're losing heat because water loss rates exceed heat gained and it's hot outside
            self.water_loss_mass = (((self.evaporative_heat_loss)/2.41)*3600)/1000.
            self.dehydration_mass -= self.water_loss_mass
            self.excess_water = ((self.energy_balance/2.41)*3600)/1000.
            self.excess_proportion_of_mass_lost = ((self.excess_water)/self.MASS)
            self.metabolic_heat_required = 0.0
            self.water_loss_required = 0.0
            self.proportion_water_loss_required = 0.0
            self.excess_metabolic_heat = 0.0
            self.water_stress = Qgen + self.evaporative_heat_loss
            self.heat_stress = 0.0
        elif self.energy_balance > 0.0 and Qgen > 0.0:#if you're losing heat because it's cold
            self.water_loss_mass = (((self.evaporative_heat_loss)/2.41)*3600)/1000.
            self.dehydration_mass -= self.water_loss_mass
            self.excess_water = 0.0
            self.excess_proportion_of_mass_lost = 0.0
            self.metabolic_heat_required = self.energy_balance
            self.water_loss_required = 0.0
            self.proportion_water_loss_required = 0.0
            self.excess_metabolic_heat = 0.0
            self.water_stress = 0.0
            self.heat_stress = Qgen - self.resting_metabolic_rate
        elif self.energy_balance < 0.0 and Qgen < 0.0:# if you're gaining heat because it's hot and water loss can't match
            self.water_loss_mass = (((self.evaporative_heat_loss)/2.41)*3600)/1000.
            self.dehydration_mass -= self.water_loss_mass
            self.excess_water = 0.0
            self.excess_proportion_of_mass_lost = 0.0
            self.metabolic_heat_required = 0.0
            self.water_loss_required = (((self.energy_balance*-1.0)/2.41)*3600)/1000.
            self.proportion_water_loss_required = (self.water_loss_required)/self.MASS
            self.excess_metabolic_heat = 0.0
            self.water_stress = Qgen + self.evaporative_heat_loss
            self.heat_stress = 0.0
        elif self.energy_balance < 0.0 and Qgen > 0.0: #if you're gaining heat but it's cold outside (maybe due to metabolic rates being high)
            self.water_loss_mass = (((self.evaporative_heat_loss)/2.41)*3600)/1000.
            self.dehydration_mass -= self.water_loss_mass
            self.excess_water = 0.0
            self.excess_proportion_of_mass_lost = 0.0
            self.metabolic_heat_required = 0.0
            self.water_loss_required = (((self.energy_balance*-1.0)/2.41)*3600)/1000.
            self.proportion_water_loss_required = (self.water_loss_required)/self.MASS
            self.excess_metabolic_heat = self.energy_balance*-1.0
            self.water_stress = 0.0
            self.heat_stress = Qgen - self.resting_metabolic_rate
        else:
            print('Error: update energy balance function -> Qgen equals energy balance')
            
    def calculate_empirical_body_conductance(self,Te):
        'a function is only used if physiological sensitivities are known, not used for the publication'
        return (self.resting_metabolic_rate - self.evaporative_heat_loss)/(self.T_b - Te)
        
    def biophysical_demand(self,Qgen):
        'a function for converting net sensible heat flux'
        if Qgen < 0:
            self.Qgen_water = Qgen * 3600.
            self.Qgen_water_sum += Qgen
            self.Qgen_heat = 0.0
        elif Qgen > 0:
            self.Qgen_heat = Qgen * 3600.
            self.Qgen_water = 0.0
        else:
            print('Error: biophysical demand function -> Qgen equals zero')
            
    def define_Tes_past(self,mins,maxes,day,hour,tau,soil_mins,soil_maxes,windspeed):
        'a function finds the standard operative temperature in the previous hour'
        if self.physiology_known == 1.0:
            if hour == 0.0:
                self.Tes_past = self.standard_operative_temperature(mins,maxes,day-1.,23.,tau,soil_mins,soil_maxes,windspeed)
                self.Qgen_past = self.Q_gen(mins,maxes,day-1.,23.,tau,soil_mins,soil_maxes,windspeed)
            else:
                pass
        else:
            if hour == 0.0:
                self.Tes_past = self.standard_operative_temperature(mins,maxes,day-1.,23.,tau,soil_mins,soil_maxes,windspeed)
                self.Qgen_past = self.Q_gen(mins,maxes,day-1.,23.,tau,soil_mins,soil_maxes,windspeed)
            else:
                pass
            
    def activity_thermal_stress(self,Tes_current,Tes_previous,species,hour):
        'a function that restricts activity based on body temperature, not used for publication'
        self.activity_above_thermal_stress = 0.0
        if Tes_previous > self.Tb_max and Tes_current > self.Tb_max:
            self.activity_above_thermal_stress = 1.0
        elif Tes_previous < self.Tb_max and Tes_current > self.Tb_max:
            slope = Tes_current - Tes_previous
            intercept = Tes_current - (slope * hour)
            threshold_time = (self.Tb_max - intercept)/slope
            self.activity_above_thermal_stress = hour - threshold_time
        elif Tes_previous > self.Tb_max and Tes_current < self.Tb_max:
            slope = Tes_current - Tes_previous
            intercept = Tes_current - (slope * hour)
            threshold_time = (self.Tb_max - intercept)/slope
            self.activity_above_thermal_stress = hour - threshold_time
        elif Tes_previous < self.Tb_max and Tes_current < self.Tb_max:
            pass
        else:
            print("Error: condition not met in activity thermal stress function")
            print(species)
        
    def activity_water_stress(self,Tes_current,Tes_previous,Qgen_past,hour):
        'a function that restricts activity based on water balance, not used for publication'
        self.activity_above_water_stress = 0.0
        if Tes_previous <= 30.0:
            previous_water_heat_ratio = self.water_heat_ratio_intercept*math.exp((self.water_heat_ratio_slope*30.0))
            previous_evaporative_heat_loss = (self.water_loss_intercept*math.exp((self.water_loss_slope*30.)))/1000.
        elif Tes_previous > 30.0:
            previous_water_heat_ratio = self.water_heat_ratio_intercept*math.exp((self.water_heat_ratio_slope*Tes_previous))
            previous_evaporative_heat_loss = (self.water_loss_intercept*math.exp((self.water_loss_slope*Tes_previous)))/1000.
        previous_resting_metabolic_rate = previous_evaporative_heat_loss/previous_water_heat_ratio
        previous_water_heat_balance = previous_evaporative_heat_loss - previous_resting_metabolic_rate
        previous_energy_balance = Qgen_past + previous_water_heat_balance
        if previous_energy_balance > 0.0 and self.energy_balance < 0.0 or previous_energy_balance < 0.0 and self.energy_balance > 0.0:
            slope = self.energy_balance - previous_energy_balance
            intercept = self.energy_balance - (slope * hour)
            threshold_time = (0.0 - intercept)/slope
            self.activity_above_water_stress = hour - threshold_time
        elif previous_energy_balance < 0.0 and self.energy_balance < 0.0:
            self.activity_above_water_stress = 1.0
        else:
            pass
            
    def seek_shade(self,air_temperature,ground_temperature,T_threshold,mins,maxes,day,hour,tau):
        'a function determines where and when birds and small mammals seek microhabitats based on environmental conditions'
        self.hS(day,hour,tau)
        if self.scenario == 'modern':
                ground_temperature = self.modern_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                air_temperature = self.modern_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                if hour == 0:
                    previous_ground_temperature = self.modern_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(23))+(self.site*288)]
                    previous_air_temperature = self.modern_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(23))+(self.site*288)]
                else:
                    previous_ground_temperature = self.modern_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour-1))+(self.site*288)]
                    previous_air_temperature = self.modern_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(hour-1))+(self.site*288)]
        elif self.scenario == 'historic':
                ground_temperature = self.historic_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                air_temperature = self.historic_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(hour))+(self.site*288)]
                if hour == 0:
                    previous_ground_temperature = self.historic_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(23))+(self.site*288)]
                    previous_air_temperature = self.historic_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(23))+(self.site*288)]
                else:
                    previous_ground_temperature = self.historic_soils.D0cm.iloc[int((int((day-15)/30) * 24) + int(hour-1))+(self.site*288)]
                    previous_air_temperature = self.historic_soils.Tair.iloc[int((int((day-15)/30) * 24) + int(hour-1))+(self.site*288)]
        if self.type == 'bird':
            if self.soil_ref == 'above':
                if air_temperature >= T_threshold:
                    slope = air_temperature - previous_air_temperature
                    intercept = air_temperature - (slope * hour)
                    threshold_time = (T_threshold - intercept)/slope
                    self.active_hours = hour - threshold_time
                    self.S = 0.5
                    self.soil_ref = 'below'
                elif air_temperature < T_threshold:
                    print('staying active1')
                    self.S = 1.0
                    self.active_hours = 0.0
            elif self.soil_ref =='below':
                if air_temperature >= T_threshold:
                    self.S = 0.5
                    self.soil_ref = 'below'
                    self.active_hours = 1.0
                elif air_temperature < T_threshold:
                    slope = air_temperature - previous_air_temperature
                    intercept = air_temperature - (slope * hour)
                    threshold_time = (T_threshold - intercept)/slope
                    self.active_hours = hour - threshold_time
                    self.S = 1.0
                    self.soil_ref = 'above'
        elif self.type == 'mammal':
            if self.activity_pattern == 'diurnal':
                if self.daylight > 0:
                    if ground_temperature >= T_threshold:
                        if self.soil_ref == 'above':
                            slope = ground_temperature - previous_ground_temperature
                            intercept = ground_temperature - (slope * hour)
                            threshold_time = (T_threshold - intercept)/slope
                            self.active_hours = hour - threshold_time
                            self.soil_ref = 'below'
                            self.S = 0.0
                        elif self.soil_ref == 'below':#if it's day, too hot an you're already below
                            self.active_hours = 1.0
                            self.S = 0.0
                    elif ground_temperature < T_threshold:#if ground temperature is fine and the previous hour was dark
                        if self.soil_ref == 'below' and previous_ground_temperature < T_threshold:#if you're below and the previous hour was too hot vs not
                            self.active_hours = 0.0
                            self.soil_ref = 'above'
                            self.S = 1.0
                        elif self.soil_ref == 'below' and previous_ground_temperature > T_threshold:
                            slope = ground_temperature - previous_ground_temperature
                            intercept = ground_temperature - (slope * hour)
                            threshold_time = (T_threshold - intercept)/slope
                            self.active_hours = hour - threshold_time
                            self.soil_ref = 'above'
                            self.S = 1.0
                        elif self.soil_ref == 'above' and previous_ground_temperature < T_threshold:#if you're below and the previous hour was too hot vs not
                            self.active_hours = 0.0
                            self.soil_ref = 'above'
                            self.S = 1.0
                if self.daylight == 0.0:
                    self.soil_ref = 'below'  
                    self.active_hours = 0.0                  
            elif self.activity_pattern == 'nocturnal':
                if self.daylight == 0.0:
                    if self.soil_ref == 'above':
                        if ground_temperature >= T_threshold and previous_ground_temperature < T_threshold:
                            slope = ground_temperature - previous_ground_temperature
                            intercept = ground_temperature - (slope * hour)
                            threshold_time = (T_threshold - intercept)/slope
                            self.active_hours = hour - threshold_time
                            self.soil_ref = 'below'
                            self.S = 0.0
                        elif ground_temperature >= T_threshold and previous_ground_temperature > T_threshold:
                            self.active_hours = 1.0
                            self.soil_ref = 'below'
                            self.S = 0.0
                        elif ground_temperature < T_threshold:
                            self.soil_ref = 'above'
                            self.active_hours = 0.0
                            self.S = 1.0
                    elif self.soil_ref == 'below':
                        if ground_temperature < T_threshold and previous_ground_temperature > T_threshold:
                            slope = ground_temperature - previous_ground_temperature
                            intercept = ground_temperature - (slope * hour)
                            threshold_time = (T_threshold - intercept)/slope
                            self.active_hours = hour - threshold_time
                            self.soil_ref = 'above'
                            self.S = 1.0
                        elif ground_temperature > T_threshold:
                            self.soil_ref = 'below'
                            self.active_hours = 1.0
                            self.S = 0.0
                elif self.daylight > 0.0:
                    self.soil_ref = 'below'
                    self.active_hours = 0.0
                    self.S = 0.0
                                            
    def find_sun(self,day,hour,tau,sun_state):
        'a function that determines whether the sun has risen or set'
        times = numpy.arange(hour-1,hour+0.01,0.01)
        for i in range(len(times)):
            sun = self.hS(day,times[i],tau)
            if sun_state == 'sunrise':
                if sun > 0:
                    return times[i]
            elif sun_state == 'sunset':
                if sun == 0:
                    return times[i]
         
#-----------------------------#
#----------DEFINITIONS--------#
#-----------------------------#

def generate_mass(mass):
    'a function can be used to assess the sensitivity of the results to mass'
    mass_change =  random.array((random.uniform(0,10)/100,random.uniform(0,10)/100,random.uniform(0,10)/100,random.uniform(0,10)/100,random.uniform(0,10)/100,random.uniform(10,20)/100,random.uniform(10,20)/100,random.uniform(10,20)/100,random.uniform(10,20)/100,random.uniform(10,20)/100,random.uniform(20,30)/100,random.uniform(20,30)/100,random.uniform(20,30)/100,random.uniform(20,30)/100,random.uniform(20,30)/100))
    masses = itertools.repeat(mass,15)
    mass_list = masses-(masses*mass_change)
    change = mass_change
    mass_list = numpy.insert(mass_list,0,mass)
    change = numpy.insert(change,0,0)
    return mass_list,change

def generate_seek():
    'a function can be used to assess the sensitivity of the results to the temperature at which the animal seeks microhabitats'
    seek_temperatures =  numpy.array((random.uniform(35,40),random.uniform(35,40),random.uniform(35,40),random.uniform(35,40),random.uniform(35,40),random.uniform(30,35),random.uniform(30,35),random.uniform(30,35),random.uniform(30,35),random.uniform(30,35),random.uniform(25,30),random.uniform(25,30),random.uniform(25,30),random.uniform(25,30),random.uniform(25,30)))
    seek_temperatures = numpy.insert(seek_temperatures,0,35.0)
    return list(seek_temperatures)
    
def read_microclimates(Tsoil):
        'a function reads csv files that are the outputs of NicheMapR'
        soil_df = pandas.DataFrame()
        list_of_Tsoil = glob.glob(Tsoil+str('*csv'))
        for i in range(len(list_of_Tsoil)):
            soils = pandas.read_csv(list_of_Tsoil[i])
            soils['site'] = list_of_Tsoil[i]
            soil_df = soil_df.append(soils)
        soil_df['site'] = soil_df['site'].map(lambda x: x.lstrip(Tsoil).rstrip('.csv'))
        soil_df = soil_df.drop(['Unnamed: 0'], axis = 1)
        soil_df = soil_df[['site','JULDAY','TIME','D0cm','D2.5cm','D5cm','D10cm','D15cm','D20cm','D30cm','D50cm','D100cm','D200cm','Tair','Tair_mammals']]
        soil_df['number'] = soil_df.site.str.extract('(\d+)',expand = False)
        soil_df['number'] = soil_df['number'].apply(float)
        soil_df = soil_df.sort_values(['number','JULDAY'])
        return(soil_df)

#--------------#
#  DATAFRAMES  #
#--------------#
'See code description above for explanation'
species = pandas.read_csv('path/endotherm_properties.csv')
locations = pandas.read_csv('path/mammal_sites.csv')
modern_soils = read_microclimates('path/modern_shade50/')
historic_soils = read_microclimates('path/historic_shade50/')


#-----------------#
#    SIMULATION   #
#-----------------#

def run_endotherm(site):
        'a function that coordinates with the other functions to run the full simulation'
        hourly_results = pandas.DataFrame(columns = ['type','species','activity_pattern','site','version','climate_scenario','mass_change','seek_temperature','wind','posture','fiber_density','shade','feather_depths','orientation','shape','physiology_known', 'significant', 'elevational_preference','habitat','diet','migratory','mass','surface_area','occupancy_decline','julian_day','hour','Srad','Rabs','Remi','Tes','Tref','soil_ref','Te_dorsal','Hi_d','Ri_d','Ks','Kf_d','Ksfi_d','Ke_d','Ke_empirical','Ke_theoretical','Tb','Qgen','EHL','MHP','EHL/MHP','energy_balance','water_loss_mass','dehydration_mass','excess_proportion_of_mass_lost','excess_water','metabolic_heat_required','water_loss_required','proportion_water_loss_required','excess_metabolic_heat','Qgen_water_avg','Qgen_water_sum','Qgen_heat_avg','Qgen_heat_sum','activity_thermal_stress','activity_water_stress','water_stress','heat_stress','hr_restriction'])
        windspeeds = [0.1]#[0.1,1.0,2.0,3.0]
        posture = [0.0]#for azimuth function, [-0.5,-0.25,0.0,0.25,0.5]
        fiber_density = [0.0]#for self.fiber_number, [-0.5,-0.25,0.0,0.25,0.5]
        shade_list = [0.0] #for self.S,[0.0,-0.25,-0.5]
        feather_depth = [0.0]#for insulation_depth, [-0.5,-0.25,0.0,0.25,0.5]
        orientation = [0.0]#for view_factor,[0.0,-0.25,-0.5,-0.75]
        shape = [0.0]#for length, [-0.5,-0.25,0.0,0.25,0.5]
        climates = ['modern','historic']
        for spp in range(len(species)):
            mass_list = [[species.mass[spp]],[0]]
            for climate_scenario in range(len(climates)):
                mins = [0]
                maxes = [0]
                soil_mins = [0]
                soil_maxes = [0]
                for each_mass in range(len(mass_list[0])):
                    seek_temperatures = [35.0]
                    julian_day = [106,136,167]
                    for each_seek in range(len(seek_temperatures)):
                        for winds in range(len(windspeeds)):
                            for postures in range(len(posture)):
                                for densities in range(len(fiber_density)):
                                    for shades in range(len(shade_list)):
                                        for depths in range(len(feather_depth)):
                                            for orientations in range(len(orientation)):
                                                for shapes in range(len(shape)):
                                                    endotherm = Individual(species.type[spp],climates[climate_scenario],species.common_name[spp],locations.latitude.iloc[site],locations.longitude_normal.iloc[site],locations.elevation.iloc[site],species.length[spp],species.width[spp],species.height[spp],species.insulation_length_dorsal[spp],species.insulation_length_ventral[spp],species.density_of_fibers[spp],species.insulation_depth_dorsal[spp],species.insulation_depth_ventral[spp],species.physiology_known[spp],species.body_temperature_min[spp],species.body_temperature_max[spp],species.lower_critical_temperature[spp],species.upper_critical_temperature[spp],mass_list[0][each_mass],species.shortwave_absorptance_dorsal[spp],species.shortwave_absorptance_ventral[spp],species.longwave_absorptance[spp],species.water_heat_ratio_slope[spp],species.water_heat_ratio_intercept[spp],species.water_loss_slope[spp],species.water_loss_intercept[spp],species.body_temperature_slope1[spp],species.body_temperature_slope2[spp],species.body_temperature_intercept[spp],0.1,posture[postures],fiber_density[densities],shade_list[shades],feather_depth[depths],orientation[orientations],mass_list[1][each_mass],species.water_threshold[spp],species.insulation_conductivity[spp],species.activity_pattern[spp],site,modern_soils,historic_soils,locations.soil.iloc[site])
                                                    for j in range(len(julian_day)):
                                                        for hour in numpy.arange(0,24,(1)):
                                                            if species.physiology_known[spp] == 1.0:
                                                                endotherm.define_Tes_past(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Tes = endotherm.standard_operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                endotherm.update_Tb(Tes,species.body_temperature_max[spp])
                                                                Tes = endotherm.standard_operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                endotherm.update_skin_conductance(Tes)
                                                                Tes = endotherm.standard_operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Te = endotherm.operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Qgen = endotherm.Q_gen(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                endotherm.update_water_loss_rate(Tes)
                                                                endotherm.update_water_heat_ratio(Tes)
                                                                endotherm.update_water_heat_balance()
                                                                endotherm.update_energy_balance(Qgen)
                                                                Ke_empirical = endotherm.calculate_empirical_body_conductance(Te)
                                                                Ke_theoretical = endotherm.Ke_overall(mins,maxes,julian_day[j],hour,soil_mins,soil_maxes,windspeeds[winds])#/abs(endotherm.T_b - Te)#why absolute value here?
                                                                endotherm.biophysical_demand(Qgen)
                                                                endotherm.activity_thermal_stress(Tes,endotherm.Tes_past,species.species[spp],hour)
                                                                endotherm.activity_water_stress(Tes,endotherm.Tes_past,endotherm.Qgen_past,hour)
                                                                Hi_d = endotherm.convective_conductance(mins,maxes,julian_day[j],hour,soil_mins,soil_maxes,windspeeds[winds])/(endotherm.surface_area_outer/2.0)#W m-2 C-1
                                                                Ri_d = endotherm.radiative_conductance(mins,maxes,julian_day[j],hour,soil_mins,soil_maxes)/(endotherm.surface_area_outer/2.0)#W m-2 C-1
                                                                Ksfi_d = (endotherm.conductance_skin * endotherm.conductance_insulation_dorsal)/(endotherm.conductance_skin + endotherm.conductance_insulation_dorsal)#W m-2 K-1
                                                            else:
                                                                endotherm.seek_shade(endotherm.air_temp(mins,maxes,julian_day[j],hour),endotherm.ground_temp_NicheMapR(julian_day[j],hour),seek_temperatures[each_seek],mins,maxes,julian_day[j],hour,TAU)
                                                                Tes = endotherm.standard_operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                endotherm.update_skin_conductance(Tes)
                                                                Tes = endotherm.standard_operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Srad = endotherm.hS(julian_day[j],hour,TAU)
                                                                Rabs = endotherm.radiation_abs('dorsal',mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Remi = endotherm.E_S*STEFAN_BOLTZMANN*((273.5 + endotherm.Tref)**4)
                                                                Te = endotherm.operative_temperature(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Qgen = endotherm.Q_gen(mins,maxes,julian_day[j],hour,TAU,soil_mins,soil_maxes,windspeeds[winds])
                                                                Ke_empirical = endotherm.calculate_empirical_body_conductance(Te)
                                                                Ke_theoretical = endotherm.Ke_overall(mins,maxes,julian_day[j],hour,soil_mins,soil_maxes,windspeeds[winds])/abs(endotherm.T_b - Te)
                                                                endotherm.biophysical_demand(Qgen)
                                                                Hi_d = endotherm.convective_conductance(mins,maxes,julian_day[j],hour,soil_mins,soil_maxes,windspeeds[winds])
                                                                Ri_d = endotherm.radiative_conductance(mins,maxes,julian_day[j],hour,soil_mins,soil_maxes)
                                                                Ksfi_d = (endotherm.conductance_skin * endotherm.conductance_insulation_dorsal)/(endotherm.conductance_skin + endotherm.conductance_insulation_dorsal)
                                                            dataframe = pandas.DataFrame([[species.type[spp],species.common_name[spp],endotherm.activity_pattern,endotherm.site + 1,species.version[spp],climates[climate_scenario],mass_list[0][each_mass],seek_temperatures[each_seek],windspeeds[winds],posture[postures],fiber_density[densities],shade_list[shades],feather_depth[depths],orientation[orientations],shape[shapes],species.physiology_known[spp],species.mass[spp],endotherm.surface_area_outer,julian_day[j],hour,Srad,Rabs,Remi,Tes,endotherm.Tref,endotherm.soil_ref,endotherm.dorsal_Te,Hi_d,Ri_d,endotherm.conductance_skin,endotherm.conductance_insulation_dorsal,Ksfi_d,endotherm.Ke_d,Ke_empirical,Ke_theoretical,endotherm.T_b,Qgen,endotherm.evaporative_heat_loss,endotherm.resting_metabolic_rate,endotherm.water_heat_balance,endotherm.energy_balance,endotherm.water_loss_mass,endotherm.dehydration_mass,endotherm.excess_proportion_of_mass_lost,endotherm.excess_water,endotherm.metabolic_heat_required,endotherm.water_loss_required,endotherm.proportion_water_loss_required,endotherm.excess_metabolic_heat,endotherm.Qgen_water,endotherm.Qgen_water,endotherm.Qgen_heat,endotherm.Qgen_heat,endotherm.activity_above_thermal_stress,endotherm.activity_above_water_stress,endotherm.water_stress,endotherm.heat_stress,endotherm.active_hours]],columns = ['type','species','activity_pattern','site','version','climate_scenario','mass_change','seek_temperature','wind','posture','fiber_density','shade','feather_depths','orientation','shape','physiology_known','mass','surface_area','julian_day','hour','Srad','Rabs','Remi','Tes','Tref','soil_ref','Te_dorsal','Hi_d','Ri_d','Ks','Kf_d','Ksfi_d','Ke_d','Ke_empirical','Ke_theoretical','Tb','Qgen','EHL','MHP','EHL/MHP','energy_balance','water_loss_mass','dehydration_mass','excess_proportion_of_mass_lost','excess_water','metabolic_heat_required','water_loss_required','proportion_water_loss_required','excess_metabolic_heat','Qgen_water_avg','Qgen_water_sum','Qgen_heat_avg','Qgen_heat_sum','activity_thermal_stress','activity_water_stress','water_stress','heat_stress','hr_restriction'])
                                                            hourly_results = hourly_results.append(dataframe)
        hourly_results.to_csv('species_output/mammal_v6_shade50/hourly_v5_standard_site'+str(site+1)+'.csv',columns=['type','species','activity_pattern','site','version','climate_scenario','mass_change','seek_temperature','wind','posture','fiber_density','shade','feather_depths','orientation','shape','physiology_known','mass','surface_area','julian_day','hour','Srad','Rabs','Remi','Tes','Tref','soil_ref','Te_dorsal','Hi_d','Ri_d','Ks','Kf_d','Ksfi_d','Ke_d','Ke_empirical','Ke_theoretical','Tb','Qgen','EHL','MHP','EHL/MHP','energy_balance','water_loss_mass','dehydration_mass','excess_proportion_of_mass_lost','excess_water','metabolic_heat_required','water_loss_required','proportion_water_loss_required','excess_metabolic_heat','Qgen_water_avg','Qgen_water_sum','Qgen_heat_avg','Qgen_heat_sum','activity_thermal_stress','activity_water_stress','water_stress','heat_stress','hr_restriction'],index = False)


if __name__ == '__main__':
    'an if statement that multiprocesses the function above to run faster'
    pool = multiprocessing.Pool(processes = 24)
    pool.map(run_endotherm,range(0,90,1))
    pool.close()
    pool.join()
