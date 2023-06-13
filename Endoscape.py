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
import scipy.interpolate as sci
from collections import OrderedDict

#--------------------#
#-----CONSTANTS------#
#--------------------#

SOLAR_CONSTANT = 1360. #W*m^-2
TAU = 0.7 #0.7 clear day
STEFAN_BOLTZMANN = 5.670373*10**(-8) #W*m^-2*K^-4
ALBEDO = 0.30 #ground reflectance (albedo) of dry sandy soil ranges 0.25-0.45
E_G = 0.9 ##surface emissivity of sandy soil w/<2% organic matter approx 0.88
OMEGA = math.pi/12.
Rv = 461.5 #J*K^-1*kg^-1
L = 2.5*10**6 #J per kg

#-------------------#
#-----CLASSESS------#
#-------------------#

class Individual():
    def __init__(self,endotherm_type,
                 common_name,
                 latitude,
                 longitude,
                 elevation,
                 length,
                 width,
                 height,
                 insulation_length_dorsal,
                 insulation_length_ventral,
                 density_of_fibers,
                 insulation_depth_dorsal,
                 insulation_depth_ventral,
                 physiology_known,
                 Tb,
                 Tb_max,
                 LCT,
                 UCT,
                 mass,
                 shortwave_absorbance_dorsal,
                 shortwave_absorbance_ventral,
                 longwave_absorbance,
                 water_heat_ratio_slope,
                 water_heat_ratio_intercept,
                 water_loss_slope,
                 water_loss_intercept,
                 Tb_slope1,
                 Tb_slope2,
                 Tb_intercept,
                 windspeed,
                 posture,
                 fiber_density,
                 shade,
                 feather_depths,
                 orientation,
                 shape,
                 water_threshold,
                 insulation_conductivity,
                 activity_pattern,
                 site_number,
                 soil_depth):
        self.type = endotherm_type
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
        self.length = length
        self.height = height
        self.width = width
        self.A_radius = self.length/2.0 - ((length/2.0) * (shape*0.35)) #radius of length of prolate spheroid
        self.B_radius = self.width/2.0 - ((width/2.0) * (shape*0.35))#radius of width of prolate spheroid
        self.C_radius = self.height/2.0 - ((height/2.0) * (shape*0.35))#radius of height of prolate spheroid
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
        self.conductance_insulation_dorsal = self.conductivity_insulation/self.insulation_depth_dorsal#W K-1
        self.conductance_insulation_ventral = self.conductivity_insulation/self.insulation_depth_ventral#W K-1
        self.conductivity_skin = 2.8 #thermal conductivity of skin W m-1 C-1,vasoconstricted
        self.skin_depth = 0.01 * (self.D**(0.6))#Bakken 1976
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
        self.daylight = 0.0
        self.Tref = 0.0
        self.Te_dorsal = 0.0
        self.Ke_d = 0.0
        self.Hi_d = 0.0
        self.Ri_d = 0.0
        self.soil_ref = 'above'
        self.soil_depth = soil_depth
        self.site_label = 'empty'

    def orbit_correction(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        return 1 + 2 * 0.01675 * math.cos((((2*math.pi)/365))*day)
    
    def direct_solar_radiation(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        return self.orbit_correction(day)*SOLAR_CONSTANT
        
    def f(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        return 279.575 + (0.9856 * day)
    
    def ET(self,day):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        _f = math.radians(self.f(day))
        return (-104.7*math.sin(_f)+596.2*math.sin(2*_f)+4.3*math.sin(3*_f)-12.7*math.sin(4*_f)-429.3*math.cos(_f)-2.0*math.cos(2*_f)+19.3*math.cos(3*_f))/3600.
    
    def LC(self,lon):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        return ((lon%15)*4.0)/60

    def t0(self,lc,et):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        t = 12 + lc - et
        return t

    def hour(self,t,t_zero):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
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
        'optimal air mass number (eq. 11.11), a function required to estimate solar radiation from Campbell and Norman (1998)'
        p_a = 101.3*math.exp(-self.altitude/8200)
        if math.cos(self.zenith(day,hrs))>=0.:
            return p_a/(101.3*(math.cos(self.zenith(day,hrs))))
        else:
            return 0.
            
    def hS0(self,day,hrs):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
        z = self.zenith(day,hrs)
        if math.cos(z)>= 0.:
            return self.direct_solar_radiation(day)*(math.cos(z))
        else:
            return 0.
            
    def hS(self,day, hrs, tau):
        'a function required to estimate solar radiation from Campbell and Norman (1998)'
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

    def longwave_sky(self,temperature,emissivity):
        'a function for long wave radiation from the sky'
        return emissivity*(5.670373*10**-8 * (temperature + 273.15)**4)

    def sky_emissivity(self,cloud_cover):
        'a function that calculates sky emissivity'
        return ((1 - 0.84 * cloud_cover) * 0.79) + (0.84 * cloud_cover)  # pg 173 in Campbell and Norman, equation 10.12

    def longwave_ground(self,temperature):
        'a function for long wave radiation from the ground'
        return E_G*STEFAN_BOLTZMANN*(temperature+273.15)**4.

    def dimensionless_temperature(self,hour):
        'a function that calculates the dimensionless temperature'
        return 0.44-(0.46*math.sin(((math.pi/12.)*hour)+0.9))+0.11*math.sin(2.*(math.pi/12.)*hour+0.9)

    def Tair(self,Tmin,Tmax,day,hour):
        'a function that returns air temperature from monthly minimum and maximum temperatures'
        if day < 0:
            day = 364 + day
        if hour > -1.0 and hour <= 5.:
            T_air = Tmax[day-1]*self.dimensionless_temperature(hour)+Tmin[day]*(1-self.dimensionless_temperature(hour))
            self.Tref = T_air
            return T_air
        if hour > 5. and hour <= 14.:
            T_air = Tmax[day]*self.dimensionless_temperature(hour)+Tmin[day]*(1-self.dimensionless_temperature(hour))
            self.Tref = T_air
            return T_air
        if hour >14 and hour <= 25.:
            if day == 364:
                T_air = Tmax[day]*self.dimensionless_temperature(hour)+Tmin[0]*(1-self.dimensionless_temperature(hour))
                self.Tref = T_air
                return T_air
            else:
                T_air = Tmax[day]*self.dimensionless_temperature(hour)+Tmin[day+1]*(1-self.dimensionless_temperature(hour))
                self.Tref = T_air
                return T_air

    def Tground(self,T_air,T_deep,Tmin,Tmax,Titerate,RH,day,hour,n,windspeed,cloud_cover,sky_emissivity,tau):
        'a function to estimate the soil surface temperature from Leaf and Erell (2018)'
        net_shortwave_radiation = (1 - ALBEDO) * ((1 - cloud_cover) * (self.hS(day, hour,tau)) + self.diffuse_solar(day, hour, tau))
        net_longwave_radiation = self.longwave_sky(T_air,sky_emissivity)-((1-0.96)*self.longwave_ground(Titerate))
        net_radiative_flux = (net_shortwave_radiation + net_longwave_radiation)-(0.96*STEFAN_BOLTZMANN*((Titerate+273.)**4))
        air_pressure = (101325. * (1. - (2.2569 * 10 ** -5) * self.altitude) ** 5.2553)
        air_density = air_pressure / (287.04 * (T_air+273.))
        e_s = 0.611*math.exp((L/Rv)*((1./273.15)-(1./(T_air+273.))))
        e_a = (RH/100.0) * e_s
        mixing_ratio = (0.6257 * (e_a * 1000)) / (air_pressure - (1.006 * (e_a * 1000)))
        specific_heat = (1004.84 + (1846.4 * mixing_ratio)) / (1 + mixing_ratio)
        z_0 = 0.0005 #0.001, roughness length, m, Porter et al. 1973, Kelso Dunes, range 0.05 - 0.1 cm
        z_r = 2.0 #reference height, m, Porter et al. 1973
        hc = (((air_density * specific_heat * (0.4**2) * windspeed)/(math.log(((z_r/z_0)+1))**2))) #roughness length, cm, Porter et al. 1973, W m-2 C-1
        r_a = ((specific_heat * air_density) / hc) / 100 #aerodynamic resistance, s m-1
        thermal_conductivity_soil = 0.25 #dry sand, W m-1 K-1, range of thermal conductivities: 0.25 - 3.0 W m-1 K-1; Chen 2008
        bulk_density = 1200.0 # bulk density of sand soils generally varies from 1.2 – 1.4 Mg m-3.; Alnefaie and Abu-Hamdeh (2013)
        specific_heat_soil = 830.0 # specific heat of sand varies from 0.83 to 1.67 kJ kg-1 C-1 depending on the water content; Alnefaie and Abu-Hamdeh (2013)
        volumetric_heat_capacity = bulk_density * specific_heat_soil
        diffusivity_soil = thermal_conductivity_soil / volumetric_heat_capacity
        z_deep = 0.80 #max depth; m
        z_mid = 0.20 #mid depth; m
        damping_depth = math.sqrt(2.0*(diffusivity_soil/((2*math.pi)/(24.0*3600.0)))) #see example 1.2 in Campbell and Norman
        T_mid = T_deep + ((Tmax - Tmin) / 2.0) * (2.71 ** (-z_mid / damping_depth)) * math.sin((math.pi / 12.) * (hour - 8.) - (z_mid / damping_depth))
        delta_time = 1.0
        net_evaporative_flux = 0.0
        T_surface = (net_radiative_flux - net_evaporative_flux + ((air_density * specific_heat * T_air)/r_a) + ((((thermal_conductivity_soil/z_deep)*T_deep)+((volumetric_heat_capacity*z_deep)/(2*delta_time)*T_mid))/(1+(z_mid/z_deep)+((volumetric_heat_capacity*z_mid*z_deep)/(2*delta_time*thermal_conductivity_soil)))))/(((air_density * specific_heat)/r_a) + ((((thermal_conductivity_soil/z_deep))+((volumetric_heat_capacity*z_deep)/(2*delta_time)))/(1+(z_mid/z_deep)+((volumetric_heat_capacity*z_mid*z_deep)/(2*delta_time*thermal_conductivity_soil)))))
        if n < 1:
            return T_surface
            #return [T_air,T_surface,net_radiative_flux,T_deep,T_mid,tau,cloud_cover]
        else:
            return self.Tground(T_air,T_deep,Tmin,Tmax,T_surface,RH,day,hour,n-1,windspeed,cloud_cover,sky_emissivity,tau)

    def radiative_conductance(self,T_air,T_ground):
        'a function for radiative conductance'
        if self.type == "bird":
            return (4.0 * (self.surface_area_outer/2.0) * STEFAN_BOLTZMANN * self.E_S * ((T_air+273.15)**3.)) #Bakken, for bird, divided by half for 2D heat flow
        elif self.type == "mammal":
            return (4.0 * (self.surface_area_outer/2.0) * STEFAN_BOLTZMANN * self.E_S * ((T_ground+273.15)**3.)) #Bakken, for mammal, divided by half for 2D heat flow, W K-1
        else:
            print("Warning: incorrect endotherm type")
    
    def convective_conductance(self,T_air,T_ground,windspeed):
        'a function for convective conductance'
        if self.type == "bird":
            air_pressure = (101325.*(1.-(2.2569*10**-5)*self.altitude)**5.2553)
            temp_K = T_air + 273.15
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
            temp_K = T_ground + 273.15
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
    
    def radiation_abs(self,area,T_air,T_ground,day,hour,tau,windspeed,sky_emissivity):
        'a function calculates the total amount of absorbed radiation'
        if area == "dorsal" and self.type == "bird":
                return ((self.S*(self.A_S_dorsal + ((self.convective_conductance(T_air,T_ground,windspeed)+self.radiative_conductance(T_air,T_ground))/(self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_dorsal)) * (2.-self.A_S_dorsal)))*((self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS(day,hour,tau))+((self.diffuse_solar(day,hour,tau)))))+(self.A_L*((self.longwave_sky(T_air,sky_emissivity))))
        elif area == "dorsal" and self.type == "mammal":
            if self.activity_pattern == 'diurnal' and self.daylight > 0:
                return ((self.S*(self.A_S_dorsal + ((self.convective_conductance(T_air,T_ground,windspeed)+self.radiative_conductance(T_air,T_ground))/(self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_dorsal)) * (2.-self.A_S_dorsal)))*((self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS(day,hour,tau))+((self.diffuse_solar(day,hour,tau)))))+(self.A_L*((self.longwave_sky(T_air,sky_emissivity))))
            else:
                return (self.S*(self.A_S_dorsal*(self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS(day,hour,tau)))+((self.diffuse_solar(day,hour,tau))))+(self.A_L*((self.longwave_sky(T_air,sky_emissivity))))
        elif area == "ventral" and self.type == "bird":
                return ((self.S*(self.A_S_ventral + ((self.convective_conductance(T_air,T_ground,windspeed)+self.radiative_conductance(T_air,T_ground))/(self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_ventral)) * (2.-self.A_S_ventral)))*((self.reflected_radiation(day,hour,tau))))+(self.A_L*(self.longwave_ground(T_ground)))
        elif area == "ventral" and self.type == "mammal":
            if self.activity_pattern == 'diurnal' and self.daylight > 0:
                return ((self.S*(self.A_S_ventral + ((self.convective_conductance(T_air,T_ground,windspeed)+self.radiative_conductance(T_air,T_ground))/(self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))*(1./(self.probability(day,hour) * self.insulation_length_ventral)) * (2.-self.A_S_ventral)))*((self.reflected_radiation(day,hour,tau))))+(self.A_L*(self.longwave_ground(T_ground)))
            else:
                return (self.S*(self.A_S_ventral*(self.view_factor_prolate_spheroid(self.animal_angle(day,hour),self.A_radius*2,self.B_radius*2)*self.hS(day,hour,tau))+((self.reflected_radiation(day,hour,tau))))+(self.A_L*((self.longwave_ground(T_ground)))))
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
            
    def Ke_overall(self,T_air,T_ground,field_windspeed):
        'a function for total effective conductance'
        field_dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
        field_ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
        return field_dorsal_conductance + field_ventral_conductance

    def Q_gen(self,T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity):
        'a function calculates dorsal and ventral net sensible heat flux from both sides independently and then adds them together'
        if self.type == "bird":
            dorsal_Te = T_air + (((self.radiation_abs('dorsal',T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_air)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,field_windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, bird
            ventral_Te = T_air + (((self.radiation_abs('ventral',T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_air)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,field_windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, bird
            dorsal_conductance_standard = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,self.windspeed))
            ventral_conductance_standard = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,self.windspeed))
            dorsal_conductance_field = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
            ventral_conductance_field = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
            relative_conductance_dorsal = dorsal_conductance_field/dorsal_conductance_standard
            relative_conductance_ventral = ventral_conductance_field/ventral_conductance_standard
            Tes_dorsal = relative_conductance_dorsal*dorsal_Te + (1. - relative_conductance_dorsal)*self.T_b
            Tes_ventral = relative_conductance_ventral*ventral_Te + (1. - relative_conductance_ventral)*self.T_b
            Qgen_dorsal = dorsal_conductance_field*(self.T_b - Tes_dorsal)
            Qgen_ventral = ventral_conductance_field*(self.T_b - Tes_ventral)
        if self.type == "mammal":
            dorsal_Te = T_ground + (((self.radiation_abs('dorsal',T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_ground)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,field_windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, mammal
            ventral_Te = T_ground + (((self.radiation_abs('ventral',T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_ground)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,field_windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, mammal
            dorsal_conductance_standard = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,self.windspeed))
            ventral_conductance_standard = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,self.windspeed))
            dorsal_conductance_field = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
            ventral_conductance_field = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
            relative_conductance_dorsal = dorsal_conductance_field/dorsal_conductance_standard
            relative_conductance_ventral = ventral_conductance_field/ventral_conductance_standard
            Tes_dorsal = relative_conductance_dorsal*dorsal_Te + (1. - relative_conductance_dorsal)*self.T_b
            Tes_ventral = relative_conductance_ventral*ventral_Te + (1. - relative_conductance_ventral)*self.T_b
            Qgen_dorsal = dorsal_conductance_field*(self.T_b - Tes_dorsal)
            Qgen_ventral = ventral_conductance_field*(self.T_b - Tes_ventral)
        return Qgen_dorsal + Qgen_ventral
        
    def operative_temperature(self,T_air,T_ground,day,hour,tau,windspeed,sky_emissivity):
        'a function for operative temperature'
        if self.type == "bird":
            self.dorsal_Te = T_air + (((self.radiation_abs('dorsal',T_air,T_ground,day,hour,tau,windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_air)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, bird
            ventral_Te = T_air + (((self.radiation_abs('ventral',T_air,T_ground,day,hour,tau,windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_air)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, bird
            dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,windspeed))
            ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,windspeed))
            total_conductance = dorsal_conductance + ventral_conductance
            return ((dorsal_conductance * self.dorsal_Te)+(ventral_conductance * ventral_Te))/total_conductance
        elif self.type == "mammal":
            self.dorsal_Te = T_ground + (((self.radiation_abs('dorsal',T_air,T_ground,day,hour,tau,windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_ground)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, mammmal
            ventral_Te = T_ground + (((self.radiation_abs('ventral',T_air,T_ground,day,hour,tau,windspeed,sky_emissivity)*(self.surface_area_outer/2.))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + T_ground)**4)*(self.surface_area_outer/2.)))/((self.convective_conductance(T_air,T_ground,windspeed))+self.radiative_conductance(T_air,T_ground))) #Bakken, mammal
            dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,windspeed))
            ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,windspeed))
            total_conductance = dorsal_conductance + ventral_conductance
            return ((dorsal_conductance * self.dorsal_Te)+(ventral_conductance * ventral_Te))/total_conductance
        else:
            print("Error in operative temperature")
    
    def standard_operative_temperature(self,T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity,n):
        'a function for standard operative temperature'
        dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,self.windspeed))
        ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,self.windspeed))
        standard_overall_conductance = dorsal_conductance + ventral_conductance
        field_dorsal_conductance = self.effective_conductance("dorsal",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
        field_ventral_conductance = self.effective_conductance("ventral",self.radiative_conductance(T_air,T_ground),self.convective_conductance(T_air,T_ground,field_windspeed))
        field_total_conductance = field_dorsal_conductance + field_ventral_conductance
        relative_conductance = field_total_conductance/standard_overall_conductance
        Tes = self.T_b - relative_conductance*(self.T_b - self.operative_temperature(T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity))
        self.update_skin_conductance(Tes)
        if n < 1:
            return Tes
        else:
            return self.standard_operative_temperature(T_air,T_ground,day,hour,tau,field_windspeed,sky_emissivity,n-1)

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

    def update_Tb(self,day,hour,tau):
        sun = self.hS(day, hour, tau)
        if sun > 0.0:
            self.T_b = 42.0
            self.A_radius = self.length / 2.0  # radius of length of prolate spheroid
            self.B_radius = self.width / 2.0  # radius of width of prolate spheroid
            self.C_radius = self.height / 2.0  # radius of height of prolate spheroid
            self.A_radius_insulation = self.A_radius
            self.B_radius_insulation = self.B_radius
            self.C_radius_insulation = self.C_radius
            self.surface_area_outer = (4 * math.pi * (((((self.A_radius_insulation * self.B_radius_insulation) ** 1.6) + ((self.A_radius_insulation * self.C_radius_insulation) ** 1.6) + ((self.B_radius_insulation * self.C_radius_insulation) ** 1.6)) / 3.0) ** (1 / 1.6)))  # *1.3
            self.surface_area_inner = 1.23 * self.surface_area_outer
            self.conductance_skin_insulation_dorsal = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))
            self.conductance_skin_insulation_ventral = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))
        else:
            self.T_b = 39.0
            self.A_radius = self.length / 2.0  * 0.80 # radius of length of prolate spheroid
            self.B_radius = self.width / 2.0  * 0.80 # radius of width of prolate spheroid
            self.C_radius = self.height / 2.0  * 0.80 # radius of height of prolate spheroid
            self.A_radius_insulation = self.A_radius
            self.B_radius_insulation = self.B_radius
            self.C_radius_insulation = self.C_radius
            self.surface_area_outer = (4 * math.pi * (((((self.A_radius_insulation * self.B_radius_insulation) ** 1.6) + ((self.A_radius_insulation * self.C_radius_insulation) ** 1.6) + ((self.B_radius_insulation * self.C_radius_insulation) ** 1.6)) / 3.0) ** (1 / 1.6)))  # *1.3
            self.surface_area_inner = 1.23 * self.surface_area_outer
            self.conductance_skin_insulation_dorsal = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_dorsal*(self.surface_area_outer/2.0)))
            self.conductance_skin_insulation_ventral = ((self.conductance_skin*(self.surface_area_inner/2.0)) * (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))/((self.conductance_skin*(self.surface_area_inner/2.0)) + (self.conductance_insulation_ventral*(self.surface_area_outer/2.0)))
         
#-----------------------------#
#----------DEFINITIONS--------#
#-----------------------------#
    
def read_temperature(Tair_df):
        'a function reads a csv file to generate daily minimum and maximum air temperatures'
        final_df = pandas.DataFrame()
        for i in range(len(Tair_df)):
            maxTemps = numpy.array([Tair_df['tmax1'][i] - 0.5, Tair_df['tmax1'][i], Tair_df['tmax2'][i], Tair_df['tmax3'][i], Tair_df['tmax4'][i], Tair_df['tmax5'][i], Tair_df['tmax6'][i], Tair_df['tmax7'][i], Tair_df['tmax8'][i], Tair_df['tmax9'][i], Tair_df['tmax10'][i], Tair_df['tmax11'][i], Tair_df['tmax12'][i], Tair_df['tmax12'][i]-0.5])
            minTemps = numpy.array([Tair_df['tmin1'][i] - 0.5, Tair_df['tmin1'][i], Tair_df['tmin2'][i], Tair_df['tmin3'][i], Tair_df['tmin4'][i], Tair_df['tmin5'][i], Tair_df['tmin6'][i],Tair_df['tmin7'][i], Tair_df['tmin8'][i], Tair_df['tmin9'][i], Tair_df['tmin10'][i],Tair_df['tmin11'][i], Tair_df['tmin12'][i], Tair_df['tmin12'][i] - 0.5])
            days = numpy.array([-16, 15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350, 381])
            Tmin_interpolate = sci.UnivariateSpline(days, minTemps, k=3)
            Tmax_interpolate = sci.UnivariateSpline(days, maxTemps, k=3)
            Tmax = Tmax_interpolate(numpy.arange(365))
            Tmin = Tmin_interpolate(numpy.arange(365))
            site_name = Tair_df['site'][i]
            final_df[str(site_name)+'_Tmax'] = Tmax
            final_df[str(site_name) + '_Tmin'] = Tmin
        return(final_df)

#--------------#
#  DATAFRAMES  #
#--------------#
'See code description above for explanation'
species = pandas.read_csv('path/bird_properties.csv')
locations = pandas.read_csv('path/sites_Endoscape.csv')
Tair_df = read_temperature(locations)

#-----------------#
#    SIMULATION   #
#-----------------#

def run_endotherm(species,sites,Tairs):
        'a function that coordinates with the other functions to run the full simulation'
        hourly_results = pandas.DataFrame(columns = ['type','species','feather_phenotype','mass_phenotype','activity_pattern','site','wind','posture','fiber_density','shade','feather_depths','orientation','shape','mass','surface_area','doy','hour','Srad','Rabs','Remi','Tes','Tref','soil_ref','Te_dorsal','Hi_d','Ri_d','Ks','Kf_d','Ksfi_d','Ke_d','Ke_theoretical','Tb','Qgen','Cooling_costs_avg','Cooling_costs_sum','Heating_costs_avg','Heating_costs_sum'])
        windspeeds = [0.1]#[0.1,1.0,2.0,3.0]
        posture = [0.0]#for azimuth function, [-0.5,-0.25,0.0,0.25,0.5]
        fiber_density = [0.0]#for self.fiber_number, [-0.5,-0.25,0.0,0.25,0.5]
        shade_list = [0.0] #for self.S,[0.0,-0.25,-0.5]
        feather_depth = [0.0]#for insulation_depth, [-0.5,-0.25,0.0,0.25,0.5]
        orientation = [0.0]#for view_factor,[0.0,-0.25,-0.5,-0.75]
        shape = [0.0]#for length, [-0.5,-0.25,0.0,0.25,0.5]
        for spp in range(len(species)):
            doy = [15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]
            for each_site in range(len(sites)):
                for winds in range(len(windspeeds)):
                    for postures in range(len(posture)):
                        for densities in range(len(fiber_density)):
                            for shades in range(len(shade_list)):
                                for depths in range(len(feather_depth)):
                                    for orientations in range(len(orientation)):
                                        for shapes in range(len(shape)):
                                            endotherm = Individual(species.type[spp],species.common_name[spp],sites.latitude.iloc[each_site],sites.longitude.iloc[each_site],sites.elevation.iloc[each_site],species.length[spp],species.width[spp],species.height[spp],species.insulation_length_dorsal[spp],species.insulation_length_ventral[spp],species.density_of_fibers[spp],species.insulation_depth_dorsal[spp],species.insulation_depth_ventral[spp],species.physiology_known[spp],species.body_temperature_min[spp],species.body_temperature_max[spp],species.lower_critical_temperature[spp],species.upper_critical_temperature[spp],species.mass[spp],species.shortwave_absorptance_dorsal[spp],species.shortwave_absorptance_ventral[spp],species.longwave_absorptance[spp],species.water_heat_ratio_slope[spp],species.water_heat_ratio_intercept[spp],species.water_loss_slope[spp],species.water_loss_intercept[spp],species.body_temperature_slope1[spp],species.body_temperature_slope2[spp],species.body_temperature_intercept[spp],0.1,posture[postures],fiber_density[densities],shade_list[shades],feather_depth[depths],orientation[orientations],shape[0],species.water_threshold[spp],species.insulation_conductivity[spp],species.activity_pattern[spp],sites['site'][each_site],1.0)
                                            for j in range(len(doy)):
                                                for hour in numpy.arange(0,24,(1)):
                                                    cloud_cover = 0.0
                                                    Tmin = Tairs[str(sites.site[each_site])+'_Tmin']
                                                    Tmax = Tairs[str(sites.site[each_site]) + '_Tmax']
                                                    T_air = endotherm.Tair(Tmin,Tmax,doy[j],hour)
                                                    T_deep = (Tmin[doy[j]] + Tmax[doy[j]])/2.0
                                                    tau = 0.75 - (0.35 * cloud_cover)  # pg 173 in Campbell and Norman 1998
                                                    sky_emissivity = endotherm.sky_emissivity(cloud_cover)
                                                    T_ground = endotherm.Tground(T_air,T_deep,Tmin[doy[j]],Tmax[doy[j]],T_air,10.0,doy[j],hour,5,windspeeds[winds],cloud_cover,sky_emissivity,tau)
                                                    endotherm.update_Tb(doy[j], hour, tau)
                                                    Tes = endotherm.standard_operative_temperature(T_air,T_ground,doy[j],hour,TAU,windspeeds[winds],sky_emissivity,5)
                                                    Srad = endotherm.hS(doy[j],hour,tau)
                                                    Rabs = endotherm.radiation_abs('dorsal',T_air,T_ground,doy[j],hour,tau,windspeeds[winds],sky_emissivity)
                                                    Remi = endotherm.E_S*STEFAN_BOLTZMANN*((273.5 + endotherm.Tref)**4)
                                                    Te = endotherm.operative_temperature(T_air,T_ground,doy[j],hour,tau,windspeeds[winds],sky_emissivity)
                                                    Qgen = endotherm.Q_gen(T_air,T_ground,doy[j],hour,tau,windspeeds[winds],sky_emissivity)
                                                    Ke_theoretical = endotherm.Ke_overall(T_air,T_ground,windspeeds[winds])/abs(endotherm.T_b - Te)
                                                    endotherm.biophysical_demand(Qgen)
                                                    Hi_d = endotherm.convective_conductance(T_air,T_ground,windspeeds[winds])
                                                    Ri_d = endotherm.radiative_conductance(T_air,T_ground)
                                                    Ksfi_d = (endotherm.conductance_skin * endotherm.conductance_insulation_dorsal)/(endotherm.conductance_skin + endotherm.conductance_insulation_dorsal)
                                                    dataframe = pandas.DataFrame([[species.type[spp],species.common_name[spp],species.feather_phenotype[spp],species.mass_phenotype[spp],endotherm.activity_pattern,sites.site[each_site],windspeeds[winds],posture[postures],fiber_density[densities],shade_list[shades],feather_depth[depths],orientation[orientations],shape[shapes],species.mass[spp],endotherm.surface_area_outer,doy[j],hour,Srad,Rabs,Remi,Tes,endotherm.Tref,endotherm.soil_ref,endotherm.dorsal_Te,Hi_d,Ri_d,endotherm.conductance_skin,endotherm.conductance_insulation_dorsal,Ksfi_d,endotherm.Ke_d,Ke_theoretical,endotherm.T_b,Qgen,endotherm.Qgen_water,endotherm.Qgen_water,endotherm.Qgen_heat,endotherm.Qgen_heat]],columns = ['type','species','feather_phenotype','mass_phenotype','activity_pattern','site','wind','posture','fiber_density','shade','feather_depths','orientation','shape','mass','surface_area','doy','hour','Srad','Rabs','Remi','Tes','Tref','soil_ref','Te_dorsal','Hi_d','Ri_d','Ks','Kf_d','Ksfi_d','Ke_d','Ke_theoretical','Tb','Qgen','Cooling_costs_avg','Cooling_costs_sum','Heating_costs_avg','Heating_costs_sum'])
                                                    hourly_results = hourly_results.append(dataframe)
        hourly_results.to_csv('path/hourly_thermoregulatory_costs_tb.csv',columns=['type','species','feather_phenotype','mass_phenotype','activity_pattern','site','wind','posture','fiber_density','shade','feather_depths','orientation','shape','mass','surface_area','doy','hour','Srad','Rabs','Remi','Tes','Tref','soil_ref','Te_dorsal','Hi_d','Ri_d','Ks','Kf_d','Ksfi_d','Ke_d','Ke_theoretical','Tb','Qgen','Cooling_costs_avg','Cooling_costs_sum','Heating_costs_avg','Heating_costs_sum'],index = False)
        return hourly_results

# Run Endoscape
hourly = run_endotherm(species,locations,Tair_df)
