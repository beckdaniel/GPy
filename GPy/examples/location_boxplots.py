"""
Boxplots: districts: incidences/area - altitude
"""
#NOTE outliers were removed
#NOTE it would be better to scale the incidence by the population

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

#Incidences per location - Box diagrams
var_list = ['altitude','latitude','longitude']
width_list = [30,120,120]
scale = [1,100,100]
for var,wd,scl in zip(var_list,width_list,scale):
    min_ = 100000
    max_ = 0
    pb.figure()
    for district in all_districts:
        incidences = useful.ndvi_clean(district,'incidences')
        location = useful.ndvi_data(district,var)[0,0]/scl
        area = useful.ndvi_data(district,'area')[0,0]
        pb.boxplot(incidences/area,positions=[location],widths=wd)
        if min_ > location:
            min_ = location
        if max_ < location:
            max_ = location
    pb.xlabel('%s' %var)
    pb.ylabel('incidences')
    minimax = (max_ - min_)*.1
    min_ = min_ - minimax
    max_ = max_ + minimax
    pb.xlim(min_,max_)
