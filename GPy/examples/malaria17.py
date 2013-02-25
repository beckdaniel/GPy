"""
Comparison:
 - GP regression incidences_district ~ time
 - GP regression ndvi_district ~ time
 - GP regression incidences_district ~ time + ndvi

---------------------------------

dataset: ../../../playground/malaria/uganda_ndvi_20130213.dat
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

#all_stations
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_stations = malaria_data['stations']
malaria_data.close()

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_ndvi_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
malaria_data.close()

#Multioutput model
Xlist_ = []
Xlist = []
Xlist_fut = []
Ylist = []
Ylist_ = []
Ylist_fut = []
likelihoods = []
R = len(all_stations)
stations = all_stations[:R]
I = np.arange(len(stations))
for i,district in zip(I,stations):
    #data
    Y_ = useful.ndvi_clean(district,'incidences')
    X1_ = useful.ndvi_clean(district,'time')

    #cut
    last = X1_[-1,0]
    cut = X1_[X1_ < last - 360].size

    Y = Y_[:cut,:]
    Y_fut = Y_[cut:,:]

    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]
    #min2_ = X1_.min()
    #max2_ = X1_.max()

    likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    Ylist_.append(Y_)
    Ylist.append(Y)
    Ylist_fut.append(Y_fut)

    #Index time
    Xlist_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    Xlist.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    Xlist_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

#model 4
print '\nMultioutput model'

R = R
D = 1
periodic4 = GPy.kern.periodic_exponential(1)
rbf4 = GPy.kern.rbf(1)
linear4 = GPy.kern.linear(1)
bias4 = GPy.kern.bias(1)
base_white4 = GPy.kern.white(1)
white4 = GPy.kern.cor_white(base_white4,R,index=0,Dw=2)
#base4 = linear4+periodic4*rbf4 + white4
#base4 = rbf4.copy()+rbf4.copy() + bias4
base4 = periodic4*rbf4+rbf4.copy()+bias4# +bias4
kernel4 = GPy.kern.icm(base4,R,index=0,Dw=2)

Z = np.linspace(100,1400,6)[:,None]

#m4 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=True)
m4 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m4.ensure_default_constraints()

m4.constrain_positive('kappa')

if hasattr(m4,'Z'):
    m4.scale_factor=100#00
    m4.constrain_fixed('iip',m4.Z[:m4._M,1].flatten())
    m4.unconstrain('exp_var')
    m4.constrain_fixed('exp_var',1)
    #m4.unconstrain('icm_rbf_var')
    #m4.constrain_fixed('icm_rbf_var',1)
    #m4.set('exp_var',10)
    m4.set('icm_rbf_len',3.)
    m4.set('icm_rbf_var',1)
    m4.set('exp_len',10)
else:
    m4.set('exp_len',.1)
    m4.set('exp_var',10)
    #m4.set('1_len',10)
    #m4.set('2_len',3.)
    #m4.set('1_var',10)
    #m4.set('rbf_var',.5)
m4.set('W',.001*np.random.rand(R*2))
print m4
print m4.checkgrad(verbose=1)
m4.optimize()
print m4
"""
for i in range(R):
    subs = 220 + i
    #fig=pb.subplot(subs)
    fig = pb.figure()
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m4.predict(Xlist_[i])
    GPy.util.plot.gpplot(Xlist_[i][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[i][:,1],Ylist[i],'kx',mew=1.5)
    pb.plot(Xlist_fut[i][:,1],Ylist_fut[i],'rx',mew=1.5)
    if hasattr(m4,'Z'):
        _Z = m4.Z[:m4._M,1]*m4._Zstd[:,1]+m4._Zmean[:,1]
        pb.plot(_Z,np.repeat(pb.ylim()[0],m4._M),'r|',mew=1.5)
    #pb.ylim(ym_lim)
    pb.ylabel('incidences')
    #fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    pb.title(all_stations[i])
"""
#GP regressions
for district,nd in zip(all_stations,range(R)):
    #data
    Y_ = useful.ndvi_clean(district,'incidences')
    X1_ = useful.ndvi_clean(district,'time')
    X2_ = useful.ndvi_clean(district,'ndvi')

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
    print '\n', 'ndvi'
    likelihoodw1 = GPy.likelihoods.Gaussian(X2,normalize =True)

    periodicw1 = GPy.kern.periodic_exponential(1)
    rbfw1 = GPy.kern.rbf(1)
    biasw1 = GPy.kern.bias(1)
    linearw1 = GPy.kern.linear(1)
    whitew1 = GPy.kern.white(1)

    w1 = GPy.models.GP(X1, likelihoodw1, periodicw1*rbfw1+rbfw1.copy()+biasw1+whitew1, normalize_X=True)

    #w1.ensure_default_constraints() #NOTE not working for sum of rbf's
    w1.constrain_positive('var')
    w1.constrain_positive('len')
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
    pb.ylabel('ndvi')
    #pb.xlabel('time (days)')
    pb.suptitle('%s' %district)
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #trends comparison
    """
    fig=pb.subplot(224)
    Yz_ = (Y_-Y_.mean())/Y_.std()
    X2z_ = (X2_-X2_.mean())/X2_.std()
    pb.plot(X1_,Yz_,'b')
    pb.plot(X1_,X2z_,'k--',linewidth=1.5)
    pb.ylabel('Incidence / ndvi\n(standardized)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """

    #model 1
    print '\nmodel 1'
    likelihood1 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic1 = GPy.kern.periodic_exponential(1)
    rbf1 = GPy.kern.rbf(1)
    linear1 = GPy.kern.linear(1)
    white1 = GPy.kern.white(1)

    m1 = GPy.models.GP(X1, likelihood1, rbf1*periodic1+rbf1.copy()+white1, normalize_X=True)

    #m1.ensure_default_constraints() #NOTE not working for sum of rbf's
    m1.constrain_positive('var')
    m1.constrain_positive('len')
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
    mean_,var_,lower_,upper_ = m1.predict(X1_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    ylim=pb.ylim()
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #model 2
    print '\nmodel 2'
    likelihood2 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic2 = GPy.kern.periodic_exponential(1)
    rbf2 = GPy.kern.rbf(1)
    rbf2_ = GPy.kern.rbf(2)
    linear2 = GPy.kern.linear(1)
    white2 = GPy.kern.white(2)

    m2 = GPy.models.GP(XX, likelihood2, GPy.kern.kern.prod_orthogonal(rbf2,periodic2)+rbf2_+white2, normalize_X=True)

    #m2.ensure_default_constraints() #NOTE not working for sum of rbf's
    m2.constrain_positive('var')
    m2.constrain_positive('len')
    m2.set('exp_len',.1)
    m2.set('exp_var',10)
    m2.set('rbf_var',.5)
    print m2.checkgrad()
    m2.optimize()
    print m2

    fig=pb.subplot(224)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m2.predict(XX_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylim(ylim)
    pb.ylabel('incidences')
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))


    #model 4 plots
    fig=pb.subplot(222)
    mean_,var_,lower_,upper_ = m4.predict(Xlist_[nd])
    GPy.util.plot.gpplot(Xlist_[nd][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd][:,1],Ylist[nd],'kx',mew=1.5)
    pb.plot(Xlist_fut[nd][:,1],Ylist_fut[nd],'rx',mew=1.5)
    if hasattr(m4,'Z'):
        _Z = m4.Z[:m4._M,1]*m4._Zstd[:,1]+m4._Zmean[:,1]
        pb.plot(_Z,np.repeat(pb.ylim()[0],m4._M),'r|',mew=1.5)
    pb.ylim(ym_lim)
    pb.ylabel('incidences')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

