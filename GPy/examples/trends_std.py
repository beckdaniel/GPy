"""
Time series comparison: incidences - weather variables by district
Only for those districts that have a weather station
"""
#NOTE ndvi data is avialable for all the districts

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
import sys

#My functions
sys.path.append('../../../playground/malaria')
import useful

pb.ion()
pb.close('all')

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
all_stations = malaria_data['stations']
malaria_data.close()
"""
#standardized variables vs time
#var_list_list=[['temperature_min'],['rain'],['humidity_06'],['ndvi']]
#name_list = ('temperature','rain','humidity','ndvi')
var_list_list=[['temperature_min'],['rain'],['ndvi']]
name_list = [var[0] for var in var_list_list]
subplots = np.arange(len(var_list_list)) + 100*len(var_list_list) + 11#(411,412,413,414)
lag_list = (0,0,0,0)
for district in all_stations:
    longitude,latitude,altitude,area = useful.geo(district)
    incidence,time_i = useful.filtered(district,'incidence',width=2.5,rm_zero=True)
    #incidence,time_i = useful.raw(district,'incidence')
    incidence = (incidence - incidence.mean())/incidence.std()
    pb.figure()
    for var_list,subs,name,lag in zip(var_list_list,subplots,name_list,lag_list):
        for var in var_list:
            fig = pb.subplot(subs)
            weather,time_w = useful.filtered(district,var,width=3,rm_zero=False)
            #weather,time_w = useful.raw(district,var)
            weather = (weather - weather.mean())/weather.std()
            pb.plot(time_i-lag,incidence,'r')
            pb.plot(time_w,weather,'k--',linewidth=1)
            pb.ylabel('%s' %name)
            fig.yaxis.set_major_locator(pb.MaxNLocator(3))
            fig.axes.get_xaxis().set_visible(False)
        #pb.text(2,pb.ylim()[1],'lag %s' %lag)
        pb.xlim(0,1800)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)')
    pb.suptitle('%s (altitude: %s)' %(district,altitude))
"""


#standardized-averaged variables vs time
#var_list_list=[['temperature_min'],['rain'],['humidity_06'],['ndvi']]
#name_list = ('temperature','rain','humidity','ndvi')
var_list_list=[['temperature_min'],['rain'],['ndvi']]
name_list = [var[0] for var in var_list_list]
subplots = np.arange(len(var_list_list)) + 100*len(var_list_list) + 11#(411,412,413,414)
lag_list = (0,0,0,0)
for district in all_stations:
    longitude,latitude,altitude,area = useful.geo(district)
    incidence,time_i = useful.filtered(district,'incidence',width=2,rm_zero=True)
    #incidence,time_i = useful.raw(district,'incidence')
    incidence = (incidence - incidence.mean())/incidence.std()
    pb.figure()
    for var_list,subs,name,lag in zip(var_list_list,subplots,name_list,lag_list):
        for var in var_list:
            fig = pb.subplot(subs)
            if var in ['temperature_min','temperature_max','rain','humidity_06','humidity_12']:
                weather,time_w = useful.moving_av(district,var,len=10,width=2,rm_zero=False)
            else:
                weather,time_w = useful.filtered(district,var,width=3,rm_zero=False)
            #weather,time_w = useful.raw(district,var)
            weather = (weather - weather.mean())/weather.std()
            pb.plot(time_i-lag,incidence,'r')
            pb.plot(time_w,weather,'k--',linewidth=2)
            pb.ylabel('%s' %name)
            fig.yaxis.set_major_locator(pb.MaxNLocator(3))
            fig.axes.get_xaxis().set_visible(False)
        #pb.text(2,pb.ylim()[1],'lag %s' %lag)
        pb.xlim(0,1800)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)')
    pb.suptitle('%s (altitude: %s)' %(district,altitude))

