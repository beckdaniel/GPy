#NOTE BROKEN
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
import sys

#My functions
sys.path.append('../../../playground/malaria')
import useful

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

#Forcast
#all_stations2 = all_stations[:5] + all_stations[6:]
all_stations2 = ['Kampala']
upper_lim = [12,12,12,8,14,14,12,5,7,12,5,14]
upper = 12
#for district in all_stations:
for district in all_stations2:
    #data
    X2_name = 'ndvi'
    Y_ = useful.ndvi_clean(district,'incidences')
    X1_ = useful.ndvi_clean(district,'time')
    X2_ = useful.ndvi_clean(district,X2_name)

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
    """
    print '\n', X2_name
    likelihoodw1 = GPy.likelihoods.Gaussian(X2,normalize =True)

    periodicw1 = GPy.kern.periodic_exponential(1)
    rbfw1 = GPy.kern.rbf(1)
    linearw1 = GPy.kern.linear(1)
    biasw1 = GPy.kern.bias(1)
    whitew1 = GPy.kern.white(1)

    #w1 = GPy.models.GP(X1, likelihoodw1, linearw1+periodicw1*rbfw1+biasw1+whitew1, normalize_X=True)
    w1 = GPy.models.GP(X1, likelihoodw1, rbfw1.copy()+periodicw1*rbfw1.copy()+rbfw1.copy()+whitew1+biasw1, normalize_X=True)

    w1.ensure_default_constraints()
    print w1.checkgrad()
    w1.set('exp_len',.1)
    w1.set('exp_var',10)
    w1.set('rbf_var',.5)
    w1.optimize()
    print w1

    fig=pb.subplot(235)
    min1_ = X1_.min()
    max1_ = X1_.max()
    X1_star = np.linspace(min1_,max1_,200)[:,None]
    mean_,var_,lower_,upper_ = w1.predict(X1_star)
    GPy.util.plot.gpplot(X1_star,mean_,lower_,upper_)
    pb.plot(X1,X2,'kx',mew=1.5)
    pb.plot(X1_fut,X2_fut,'rx',mew=1.5)
    yw_lim = pb.ylim()
    pb.ylabel(X2_name)
    pb.xlabel('time (days)')
    pb.suptitle('%s' %district)
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #trends comparison
    fig=pb.subplot(234)
    Yz_ = (Y_-Y_.mean())/Y_.std()
    X2z_ = (X2_-X2_.mean())/X2_.std()
    pb.plot(X1_,Yz_,'b')
    pb.plot(X1_,X2z_,'k--',linewidth=1.5)
    pb.ylabel('Incidence / %s\n(standardized)' %X2_name)
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    pb.xlabel('time (days)')

    #model 1
    print '\nmodel 1'
    likelihood1 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic1 = GPy.kern.periodic_exponential(1)
    rbf1 = GPy.kern.rbf(1)
    linear1 = GPy.kern.linear(1)
    white1 = GPy.kern.white(1)
    bias1 = GPy.kern.bias(1)

    #m1 = GPy.models.GP(X1, likelihood1, linear1+periodic1*rbf1+bias1+white1, normalize_X=True)
    m1 = GPy.models.GP(X1, likelihood1, rbf1.copy()+periodic1*rbf1.copy()+rbf1.copy()+bias1+white1, normalize_X=True)

    m1.ensure_default_constraints()
    m1.set('exp_len',.1)
    m1.set('exp_var',10)
    #m1.set('rbf_var',.5)
    m1.set('rbf_1_var',.5)
    print m1.checkgrad()
    m1.optimize()
    print m1

    #pb.figure()
    fig=pb.subplot(231)
    min1_ = X1_.min()
    max1_ = X1_.max()
    mean_,var_,lower_,upper_ = m1.predict(X1_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    #pb.ylim(0,upper*1000)
    ym_lim = pb.ylim()
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """

    #model 2
    """
    print '\nmodel 2'
    likelihood2 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic2 = GPy.kern.periodic_exponential(1)
    rbf2 = GPy.kern.rbf(1)
    rbf2_ = GPy.kern.rbf(2)
    linear2 = GPy.kern.linear(1)
    white2 = GPy.kern.white(2)
    bias2 = GPy.kern.bias(2)

    #m2 = GPy.models.GP(XX, likelihood2, GPy.kern.kern.prod_orthogonal(linear2,periodic2*rbf2)+white2, normalize_X=True)
    m2 = GPy.models.GP(XX, likelihood2, GPy.kern.kern.prod_orthogonal(rbf2.copy(),periodic2*rbf2.copy())+rbf2_.copy()+white2, normalize_X=True)

    m2.ensure_default_constraints()
    m2.set('exp_len',.1)
    m2.set('exp_var',5)
    m2.set('rbf_var',.5)
    print m2.checkgrad()
    m2.optimize()
    print m2

    fig=pb.subplot(232)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m2.predict(XX_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylim(ym_lim or None)
    #pb.ylabel('incidences')
    #pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """
    #model 3
    """
    print '\nmodel 3'

    #Create likelihood
    likelihoods = []
    likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    likelihoods.append(GPy.likelihoods.Gaussian(X2,normalize=False))
    #index time
    index = [np.zeros(X1.size)[:,None], np.ones(X1.size)[:,None]]
    XI = np.vstack([np.hstack([i,X1]) for i in index])

    index = [np.zeros(X1_.size)[:,None], np.ones(X1_.size)[:,None]]
    XIm_ = np.hstack([index[0],X1_])
    XIw_ = np.hstack([index[1],X1_])

    R = 2
    D = 1
    periodic3 = GPy.kern.periodic_exponential(1)
    rbf3 = GPy.kern.rbf(1)
    linear3 = GPy.kern.linear(1)
    bias3 = GPy.kern.bias(1)
    base_white3 = GPy.kern.white(1)
    white3 = GPy.kern.cor_white(base_white3,R,index=0,Dw=2) #FIXME
    #base3 = linear3+periodic3*rbf3 + white3
    #base3 = linear3+periodic3*rbf3 +bias3
    base3 = rbf3.copy()+periodic3*rbf3.copy() +rbf3.copy() + bias3
    kernel3 = GPy.kern.icm(base3,R,index=0,Dw=2)

    m3 = GPy.models.mGP(XI, likelihoods, kernel3+white3, normalize_X=True,normalize_Y=True)

    m3.ensure_default_constraints()

    m3.constrain_positive('kappa')
    m3.set('exp_len',.1)
    m3.set('exp_var',10)
    m3.set('rbf_var',.5)
    m3.set('W',np.random.rand(R*2))

    print m3.checkgrad(verbose=1)
    m3.optimize()
    print m3

    fig=pb.subplot(233)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m3.predict(XIm_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylim(ym_lim)
    pb.ylabel('incidences')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    fig=pb.subplot(236)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m3.predict(XIw_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,X2,'kx',mew=1.5)
    pb.plot(X1_fut,X2_fut,'rx',mew=1.5)
    pb.ylim(yw_lim)
    #pb.ylabel('ndvi')
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """

    #model 4
    print '\nmodel 4'

    #Create likelihood
    likelihoods = []
    likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    likelihoods.append(GPy.likelihoods.Gaussian(X2,normalize=False))
    #index time
    index = [np.zeros(X1.size)[:,None], np.ones(X1.size)[:,None]]
    XI = np.vstack([np.hstack([i,X1]) for i in index])

    index = [np.zeros(X1_.size)[:,None], np.ones(X1_.size)[:,None]]
    XIm_ = np.hstack([index[0],X1_])
    XIw_ = np.hstack([index[1],X1_])

    R = 2
    D = 1
    periodic4 = GPy.kern.periodic_exponential(1)
    rbf4 = GPy.kern.rbf(1)
    linear4 = GPy.kern.linear(1)
    bias4 = GPy.kern.bias(1)
    base_white4 = GPy.kern.white(1)
    white4 = GPy.kern.cor_white(base_white4,R,index=0,Dw=2) #FIXME
    #base4 = linear4+periodic4*rbf4 + white4
    #base4 = linear4+periodic4*rbf4 +bias4
    base4 = rbf4.copy()+periodic4*rbf4.copy() +rbf4.copy() + bias4
    kernel4 = GPy.kern.icm(base4,R,index=0,Dw=2)

    Z = np.linspace(100,1700,6)[:,None]

    m4 = GPy.models.multioutput_GP(XI, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

    m4.ensure_default_constraints()

    m4.constrain_positive('kappa')
    m4.set('exp_len',.1)
    m4.set('exp_var',10)
    m4.set('rbf_var',.5)
    m4.set('W',np.random.rand(R*2))

    print m4.checkgrad(verbose=1)
    m4.optimize()
    print m4

    fig=pb.subplot(233)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m3.predict(XIm_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylim(ym_lim)
    pb.ylabel('incidences')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    fig=pb.subplot(236)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m4.predict(XIw_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,X2,'kx',mew=1.5)
    pb.plot(X1_fut,X2_fut,'rx',mew=1.5)
    #pb.ylim(yw_lim)
    #pb.ylabel('ndvi')
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
