"""
Comparison:
 - GP regression incidences_district ~ time
 - GP regression ndvi_district ~ time
 - GP regression incidences_district ~ time + ndvi

---------------------------------

dataset: ../../../playground/malaria/uganda_data_20130213.dat
"""

import numpy as np
import pylab as pb
import shelve
import GPy
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

#Forcast
#all_stations2 = all_stations[:5] + all_stations[6:]
all_stations2 = ['Kampala']
upper_lim = [12,12,12,8,14,14,12,5,7,12,5,14]
cut = 56
#for district in all_stations:
for district,upper in zip(all_stations,upper_lim):
    #data
    Y_ = useful.weekly_data(district,'incidences')
    Y = Y_[:cut,:]
    Y_fut = Y_[cut:,:]

    X1_ = useful.weekly_data(district,'time')
    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]

    X2_name = 'ndvi'
    X2_ = useful.weekly_data(district,X2_name)
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
    m2.set('exp_var',10)
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
