"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/uganda_data_20130213.dat
B matrix controls the relation between districts
Incidences are assumed to have a log-normal distribution
"""
#NOTE This is a non-sparse model

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime

pb.ion()
pb.close('all')

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
all_stations = malaria_data['stations']
malaria_data.close()

def weekly_data(district_name,variable_name):
    """
    Returns a variable in specific district form file uganda_data_20130213.dat (weekly data)
    """
    #Open data file
    malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
    all_districts = malaria_data['districts']
    all_variables = malaria_data['headers']
    all_stations = malaria_data['stations']
    #all_stations = [all_stations[0]] + all_stations[2:] #Entebbe is not in the non-weather data
    station_variables = malaria_data['headers_daily']
    #Get district number in data arrange
    district = [district_name]
    d_number = int(np.arange(len(all_districts))[np.array(all_districts) == district])
    #Get variable number in data arrange
    variable = [variable_name]
    v_number = int(np.arange(len(all_variables))[np.array(all_variables) == variable])
    v_data = malaria_data['data'][d_number][:,v_number]
    #Close data file
    malaria_data.close()

    #Remove wrong data
    if district_name == 'Mbarara':
        v_data = np.hstack([v_data[:5],v_data[6:]])
    if district_name == 'Gulu':
        v_data = np.hstack([v_data[:-2],v_data[-1]])
    return v_data[:,None]

def raw_data(district_name,variable_name):
    """
    Returns a variable in a specific district from file uganda_lag_20130213.dat
    ndvi data is obtained each 10 days
    weather data is weekly
    """
    #Open data file
    malaria_data = shelve.open('../../../playground/malaria/uganda_lag_20130213.dat',writeback=False)

    if variable_name == 'ndvi':
        all_districts = malaria_data['ndvi_districts']
        #Get district number in data arrange
        district = [district_name]
        d_number = int(np.arange(len(all_districts))[np.array(all_districts) == district])
        v_data = malaria_data['ndvi_raw'][d_number]

    else:
        all_districts = malaria_data['stations']
        all_variables = malaria_data['headers']
        #Get district number in data arrange
        district = [district_name]
        d_number = int(np.arange(len(all_districts))[np.array(all_districts) == district])
        #Get variable number in data arrange
        variable = [variable_name]
        v_number = int(np.arange(len(all_variables))[np.array(all_variables) == variable])
        v_data_ = malaria_data['weather_weekly'][d_number][:,v_number]
        v_time_ =  malaria_data['weather_weekly'][d_number][:,0]
        v_data = np.hstack([v_time_[:,None],v_data_[:,None]])

    #Close data file
    malaria_data.close()
    return v_data

def match(district_name,following_var,leading_var,lag):
    """
    For a pair of variables (leading and following) matches the dates of the following variable with the dates - lag of the leading variable
    Returns matched variables, following variable time, lag
    """
    following = np.hstack([weekly_data(district_name,'time'),weekly_data(district_name,following_var)])
    leading = raw_data(district_name,leading_var)
    leading_max = leading[:,0].max()
    index = np.arange(leading.shape[0])
    match = []

    for f in following[following[:,0] <= leading_max - lag,:]:

        if leading_var == 'ndvi':
                index2 = index[leading[:,0] <  f[0] + lag + 4]
                i = index2[leading[index2,0] > f[0] + lag - 4]

        else:
            for f in following[following[:,0] <= leading_max - lag,:]:
                i = index[leading[:,0] == f[0] + lag]

        if i >= 0:
            match.append(np.hstack([f[None,:],leading[i,:]]))

    match = np.vstack(match)
    v_data = match[:,np.array(1,3)]
    t_data = match[:,0]

    return v_data,t_data,lag

def match_z(district_name,following_var,leading_var,lag):
    """
    Like match but with standardized values
    """
    #Following_var
    Z = weekly_data(district_name,following_var)
    Z = (Z - Z.mean())/Z.std()
    following = np.hstack([weekly_data(district_name,'time'),Z])
    #Leading_var
    leading = raw_data(district_name,leading_var)
    Z = leading[:,1]
    Z = (Z - Z.mean())/Z.std()
    leading[:,1] = Z
    leading_max = leading[:,0].max()

    index = np.arange(leading.shape[0])
    match = []

    for f in following[following[:,0] <= leading_max - lag,:]:

        if leading_var == 'ndvi':
                index2 = index[leading[:,0] <  f[0] + lag + 4]
                i = index2[leading[index2,0] > f[0] + lag - 4]

        else:
            for f in following[following[:,0] <= leading_max - lag,:]:
                i = index[leading[:,0] == f[0] + lag]

        if i >= 0:
            match.append(np.hstack([f[None,:],leading[i,:]]))

    match = np.vstack(match)
    v_data = match[:,np.array(1,3)]
    t_data = match[:,0]

    return v_data,t_data,lag


#Incidences per location - Box diagrams
"""
var_list = ['altitude','latitude','longitude']
width_list = [30,120,120]
scale = [1,100,100]
for var,wd,scl in zip(var_list,width_list,scale):
    min_ = 100000
    max_ = 0
    pb.figure()
    for district in all_districts:
        incidences = weekly_data(district,'incidences')
        location = weekly_data(district,var)[0,0]/scl
        pb.boxplot(incidences,positions=[location],widths=wd)
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

"""

#Variables vs time
"""
var_list_list=[['temperature_min','temperature_max'],['rain'],['humidity_06','humidity_12'],['ndvi'],['incidences']]
subplots = (511,512,513,514,515)
name_list = ('temperature','rain','humidity','NDVI','incidences')
for district in all_stations[:-5]:
    time = weekly_data(district,'time')
    pb.figure()
    for var_list,subs,name in zip(var_list_list,subplots,name_list):
        for var in var_list:
            fig = pb.subplot(subs)
            v = weekly_data(district,var)
            pb.plot(time,v,'k')
            pb.ylabel('%s' %name)
            fig.yaxis.set_major_locator(pb.MaxNLocator(3))

            fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)')
    pb.suptitle('%s' %district)
"""

#standardized Variables vs time
var_list_list=[['temperature_min'],['rain'],['humidity_06'],['ndvi']]
subplots = (411,412,413,414)
lag_list = (77,7,14,42)
name_list = ('temperature','rain','humidity','NDVI')
for district in all_stations[:-5]:
    altitude = weekly_data(district,'altitude')[0,0]
    incidences = weekly_data(district,'incidences')
    incidences = (incidences - incidences.mean())/incidences.std()
    time = weekly_data(district,'time')
    pb.figure()
    for var_list,subs,name,lag in zip(var_list_list,subplots,name_list,lag_list):
        for var in var_list:
            fig = pb.subplot(subs)
            v_ = weekly_data(district,var)
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
"""
cut = 56
for district in all_stations:
    #data
    Y_ = weekly_data(district,'incidences')
    Y = Y_[:cut,:]
    Y_fut = Y_[cut:,:]

    X1_ = weekly_data(district,'time')
    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]

    X2_name = 'ndvi'
    X2_ = weekly_data(district,X2_name)
    X2 = X2_[:cut,:]
    X2_fut = X2_[cut:,:]

    XX_ = np.hstack([X1_,X2_])
    XX = np.hstack([X1,X2])
    XX_fut = np.hstack([X1_fut,X2_fut])

    #weather 1
    likelihoodw1 = GPy.likelihoods.Gaussian(X2,normalize =True)

    periodicw1 = GPy.kern.periodic_exponential(1)
    rbfw1 = GPy.kern.rbf(1)
    linearw1 = GPy.kern.linear(1)
    whitew1 = GPy.kern.white(1)

    w1 = GPy.models.GP(X1, likelihoodw1, linearw1+periodicw1+rbfw1+whitew1, normalize_X=True)

    w1.ensure_default_constraints()
    print '\n', district
    print X2_name
    print w1.checkgrad()
    w1.set('len',.1)
    w1.optimize()
    print w1

    pb.figure()
    pb.subplot(221)
    min1_ = X1_.min()
    max1_ = X1_.max()
    X1_star = np.linspace(min1_,max1_,200)[:,None]
    mean_,var_,lower_,upper_ = w1.predict(X1_star)
    GPy.util.plot.gpplot(X1_star,mean_,lower_,upper_)
    pb.plot(X1,X2,'kx',mew=1.5)
    pb.plot(X1_fut,X2_fut,'rx',mew=1.5)
    pb.ylabel(X2_name)
    #pb.xlabel('time (days)')
    pb.suptitle('%s' %district)

    pb.subplot(223)
    Yz_ = (Y_-Y_.mean())/Y_.std()
    X2z_ = (X2_-X2_.mean())/X2_.std()
    pb.plot(X1_,Yz_,'b')
    pb.plot(X1_,X2z_,'k--',linewidth=1.5)
    pb.ylabel('Incidence / %s\n(standardized)' %X2_name)

    #model 1
    likelihood1 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic1 = GPy.kern.periodic_exponential(1)
    rbf1 = GPy.kern.rbf(1)
    linear1 = GPy.kern.linear(1)
    white1 = GPy.kern.white(1)

    m1 = GPy.models.GP(X1, likelihood1, linear1+periodic1+rbf1+white1, normalize_X=True)

    m1.ensure_default_constraints()
    #print '\n', district
    print 'model 1'
    print m1.checkgrad()
    m1.optimize()
    #print m1

    #pb.figure()
    pb.subplot(222)
    min1_ = X1_.min()
    max1_ = X1_.max()
    #X1_star = np.linspace(min1_,max1_,200)
    mean_,var_,lower_,upper_ = m1.predict(X1_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    #pb.xlabel('time (days)')
    pb.suptitle('%s' %district)

    #model 2
    likelihood2 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic2 = GPy.kern.periodic_exponential(1)
    rbf2 = GPy.kern.rbf(2)
    linear2 = GPy.kern.linear(1)
    white2 = GPy.kern.white(2)

    m2 = GPy.models.GP(XX, likelihood2, linear2*periodic2+rbf2+white2, normalize_X=True)

    m2.ensure_default_constraints()
    #print '\n', district
    print 'model 2'
    print m2.checkgrad()
    m2.optimize()
    #print m2

    pb.subplot(224)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m2.predict(XX_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    pb.xlabel('time (days)')
    #pb.title('%s' %district)
    pb.xlabel('time (days)')
"""

