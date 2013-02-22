"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/uganda_ndvi_20130213.dat
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

#all_stations
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_stations = malaria_data['stations']
malaria_data.close()

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_ndvi_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
malaria_data.close()

def ndvi_data(district_name,variable_name):
    """
    Returns a variable in specific district form file uganda_ndvi_20130213.dat (weekly data)
    """
    #Open data file
    malaria_data = shelve.open('../../../playground/malaria/uganda_ndvi_20130213.dat',writeback=False)
    all_districts = malaria_data['districts']
    all_variables = malaria_data['headers']
    #Get district number in data arrange
    district = [district_name]
    d_number = int(np.arange(len(all_districts))[np.array(all_districts) == district])
    #Get variable number in data arrange
    variable = [variable_name]
    v_number = int(np.arange(len(all_variables))[np.array(all_variables) == variable])
    v_data = malaria_data['data'][d_number][:,v_number]
    #Close data file
    malaria_data.close()
    return v_data[:,None]

def ndvi_clean(district_name,variable_name):
    """
    Returns a variable in specific district form file uganda_ndvi_20130213.dat (weekly data) after removing wrong incidences
    """
    assert variable_name in ['ndvi','time','incidences']
    inc = ndvi_data(district_name,'incidences').flatten()
    var = ndvi_data(district_name,variable_name).flatten()
    index = np.arange(inc.size)
    mean = inc.mean()
    threshold = 2*inc.std()
    #Remove above
    clean = index[inc < mean+threshold]
    inc2 = inc[clean]
    var = var[clean]
    index2 = np.arange(clean.size)
    #Remove below
    clean2 = index2[inc2 > mean-threshold]
    var = var[clean2]
    """
    if district_name == 'Apac':
        clean = index[inc < 10000]
        var = var[clean,:]
    #Remove wrong data
    if district_name == 'Mbarara':
        v_data = np.hstack([v_data[:14],v_data[15:]])
    if district_name == 'Gulu':
        v_data = np.hstack([v_data[:-2],v_data[-1]])
    """
    return var[:,None]

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
        incidences = ndvi_clean(district,'incidences')
        location = ndvi_data(district,var)[0,0]/scl
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
var_list=['ndvi','incidences']
subplots = (211,212)
name_list = ('NDVI','incidences')
for district in all_stations:
    time = ndvi_clean(district,'time')
    pb.figure()
    for var,subs,name in zip(var_list,subplots,name_list):
        fig = pb.subplot(subs)
        v = ndvi_clean(district,var)
        pb.plot(time,v,'k')
        pb.ylabel('%s' %name)
        fig.yaxis.set_major_locator(pb.MaxNLocator(3))
        fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)')
    pb.suptitle('%s' %district)
"""
#standardized Variables vs time
"""
lag = 0
for district in all_stations:
    altitude = ndvi_data(district,'altitude')[0,0]
    incidences = ndvi_clean(district,'incidences')
    incidences = (incidences - incidences.mean())/incidences.std()
    time = ndvi_clean(district,'time')
    fig = pb.figure()

    v_ = ndvi_clean(district,'ndvi')
    v = (v_ - v_.mean())/v_.std()
    pb.plot(time-lag,incidences,'r')
    pb.plot(time,v,'k--',linewidth=2)
    pb.ylabel('incidence / NDVI\nstandardized')
    #fig.yaxis.set_major_locator(pb.MaxNLocator(3))
    pb.text(2,pb.ylim()[1],'lag %s' %lag)
    pb.xlim(0,1800)
    pb.xlabel('time (days)')
    pb.suptitle('%s (altitude: %s)' %(district,altitude))

"""
#Forcast
#all_stations2 = all_stations[:5] + all_stations[6:]
all_stations2 = ['Kampala']
upper_lim = [12,12,12,8,14,14,12,5,7,12,5,14]
upper = 12
for district in all_stations[2:]:
#for district,upper in zip(all_stations,upper_lim):
    #data
    X2_name = 'ndvi'
    Y_ = ndvi_clean(district,'incidences')
    X1_ = ndvi_clean(district,'time')
    X2_ = ndvi_clean(district,X2_name)

    #cut
    last = X1_[-1,0]
    cut = X1_[X1_ < last - 360].size

    Y = Y_[:cut,:]
    Y_fut = Y_[cut:,:]

    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]

    X2 = X2_[:cut,:]
    X2_fut = X2_[cut:,:]

    XX_ = np.hstack([X1_,X2_])
    XX = np.hstack([X1,X2])
    XX_fut = np.hstack([X1_fut,X2_fut])

    pb.figure()
    pb.suptitle('%s' %district)
    print '\n', district

    #weather 1
    print '\n', X2_name
    likelihoodw1 = GPy.likelihoods.Gaussian(X2,normalize =True)

    periodicw1 = GPy.kern.periodic_exponential(1)
    rbfw1 = GPy.kern.rbf(1)
    linearw1 = GPy.kern.linear(1)
    whitew1 = GPy.kern.white(1)

    w1 = GPy.models.GP(X1, likelihoodw1, linearw1*periodicw1*rbfw1+whitew1, normalize_X=True)

    w1.ensure_default_constraints()
    print w1.checkgrad()
    w1.set('exp_len',.1)
    w1.set('exp_var',10)
    w1.set('rbf_var',.5)
    w1.optimize()
    print w1

    fig=pb.subplot(223)
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
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #trends comparison
    fig=pb.subplot(224)
    Yz_ = (Y_-Y_.mean())/Y_.std()
    X2z_ = (X2_-X2_.mean())/X2_.std()
    pb.plot(X1_,Yz_,'b')
    pb.plot(X1_,X2z_,'k--',linewidth=1.5)
    pb.ylabel('Incidence / %s\n(standardized)' %X2_name)
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #model 1
    print '\nmodel 1'
    likelihood1 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic1 = GPy.kern.periodic_exponential(1)
    rbf1 = GPy.kern.rbf(1)
    linear1 = GPy.kern.linear(1)
    white1 = GPy.kern.white(1)

    m1 = GPy.models.GP(X1, likelihood1, linear1*periodic1*rbf1+white1, normalize_X=True)

    m1.ensure_default_constraints()
    m1.set('exp_len',.1)
    m1.set('exp_var',10)
    m1.set('rbf_var',.5)
    print m1.checkgrad()
    m1.optimize()
    print m1

    #pb.figure()
    fig=pb.subplot(221)
    min1_ = X1_.min()
    max1_ = X1_.max()
    #X1_star = np.linspace(min1_,max1_,200)
    mean_,var_,lower_,upper_ = m1.predict(X1_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    pb.ylim(0,upper*1000)
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #model 2
    print '\nmodel 2'
    likelihood2 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic2 = GPy.kern.periodic_exponential(1)
    rbf2 = GPy.kern.rbf(1)
    linear2 = GPy.kern.linear(1)
    white2 = GPy.kern.white(2)

    m2 = GPy.models.GP(XX, likelihood2, GPy.kern.kern.prod_orthogonal(linear2,periodic2*rbf2)+white2, normalize_X=True)

    m2.ensure_default_constraints()
    m2.set('exp_len',.1)
    m2.set('exp_var',5)
    m2.set('rbf_var',.5)
    print m2.checkgrad()
    m2.optimize()
    print m2

    fig=pb.subplot(222)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m2.predict(XX_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylim(0,upper*1000)
    pb.ylabel('incidences')
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
