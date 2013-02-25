"""
Time series comparison: incidences - weather variables by district
Only for those districts that have a weather variable
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

#standardized Variables vs time
var_list_list=[['temperature_min'],['rain'],['humidity_06'],['ndvi']]
subplots = (411,412,413,414)
lag_list = (77,7,14,42)
name_list = ('temperature','rain','humidity','NDVI')
for district in all_stations[:-5]:
    altitude = useful.weekly_data(district,'altitude')[0,0]
    incidences = useful.weekly_data(district,'incidences')
    incidences = (incidences - incidences.mean())/incidences.std()
    time = useful.weekly_data(district,'time')
    pb.figure()
    for var_list,subs,name,lag in zip(var_list_list,subplots,name_list,lag_list):
        for var in var_list:
            fig = pb.subplot(subs)
            v_ = useful.weekly_data(district,var)
            v = (v_ - v_.mean())/v_.std()
            pb.plot(time-lag,incidences,'r')
            pb.plot(time,v,'k--',linewidth=2)
            pb.ylabel('%s' %name)
            fig.yaxis.set_major_locator(pb.MaxNLocator(3))
            fig.axes.get_xaxis().set_visible(False)
        pb.text(2,pb.ylim()[1],'lag %s' %lag)
        pb.xlim(0,1800)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)')
    pb.suptitle('%s (altitude: %s)' %(district,altitude))
