"""
Comparison:
    - GP regression incidences_district ~ time
    - GP regression ndvi_district ~ time
    - GP regression incidences_district ~ time + ndvi
    - multioutput GP incidences_district ~ time
    - multioutput GP incidences_district ~ time + ndvi

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

all_stations = ['Mubende','Masindi','Mbarara','Kampala','Kasese']

#Multioutput models outputs/inputs
Xlist_ = []
Xlist = []
Xlist_fut = []

XXlist_ = []
XXlist = []
XXlist_fut = []

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

    likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    Ylist_.append(Y_)
    Ylist.append(Y)
    Ylist_fut.append(Y_fut)

    #Index time
    Xlist_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    Xlist.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    Xlist_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

    #Index time + ndvi
    XXlist_.append(np.hstack([np.repeat(i,XX_.shape[0])[:,None],XX_]))
    XXlist.append(np.hstack([np.repeat(i,XX.shape[0])[:,None],XX]))
    XXlist_fut.append(np.hstack([np.repeat(i,XX_fut.shape[0])[:,None],XX_fut]))

#model 4
#multioutput GP incidences_district ~ time
print '\nMultioutput model incidences ~ time'

R = R
D = 1
periodic4 = GPy.kern.periodic_exponential(1)
rbf4 = GPy.kern.rbf(1)
linear4 = GPy.kern.linear(1)
bias4 = GPy.kern.bias(1)
base_white4 = GPy.kern.white(1)
white4 = GPy.kern.cor_white(base_white4,R,index=0,Dw=2)
base4 = periodic4*rbf4+rbf4.copy()+bias4
kernel4 = GPy.kern.icm(base4,R,index=0,Dw=2)

Z = np.linspace(100,1400,5)[:,None]

#m4 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=True)
m4 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m4.ensure_default_constraints()
m4.constrain_positive('kappa')
m4.unconstrain('exp_var')
m4.constrain_fixed('exp_var',1)
if hasattr(m4,'Z'):
    m4.scale_factor=100#00
    m4.constrain_fixed('iip',m4.Z[:m4._M,1].flatten())
m4.set('exp_len',.1)
m4.set('icm_rbf_var',10)
m4.set('W',.001*np.random.rand(R*2))

print m4.checkgrad(verbose=1)
m4.optimize()
print m4

#model 5
#multioutput GP incidences_district ~ time + ndvi
print '\nMultioutput model incidences ~ time + ndvi'

R = R
D = 1
periodic5 = GPy.kern.periodic_exponential(1)
rbf5 = GPy.kern.rbf(1)
rbf5_ = GPy.kern.rbf(2)
linear5 = GPy.kern.linear(1)
bias5 = GPy.kern.bias(2)
base_white5 = GPy.kern.white(2)
white5 = GPy.kern.cor_white(base_white5,R,index=0,Dw=2)
base5 = GPy.kern.kern.prod_orthogonal(periodic5,rbf5)+rbf5_+bias5
kernel5 = GPy.kern.icm(base5,R,index=0,Dw=2)

_M = 5
Z = np.linspace(100,1400,_M)[:,None]
ndvi_stats = [X2.min(),X2.mean(),X2.max()]
_Z = []
for nstats in ndvi_stats:
    _Z.append(np.hstack([np.repeat(nstats,_M)[:,None],Z]))
Z = np.vstack(_Z)

#m5 = GPy.models.mGP(Xlist, likelihoods, kernel5+white5, normalize_Y=True)
m5 = GPy.models.multioutput_GP(XXlist, likelihoods, kernel5+white5,Z=Z, normalize_X=True,normalize_Y=True)

m5.ensure_default_constraints()
m5.constrain_positive('kappa')
m5.unconstrain('exp_var')
m5.constrain_fixed('exp_var',1)
if hasattr(m5,'Z'):
    m5.scale_factor=.1
    m5.constrain_fixed('iip',m5.Z[:m5._M,1].flatten())
m5.set('exp_len',.1)
m5.set('icm_rbf_var',10)
m5.set('W',.001*np.random.rand(R*2))

print m5.checkgrad(verbose=1)
m5.optimize()
print m5

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
    #GP regression ndvi_district ~ time
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

    fig=pb.subplot(234)
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
    fig=pb.subplot(234)
    Yz_ = (Y_-Y_.mean())/Y_.std()
    X2z_ = (X2_-X2_.mean())/X2_.std()
    pb.plot(X1_,Yz_,'b')
    pb.plot(X1_,X2z_,'k--',linewidth=1.5)
    pb.ylabel('Incidence / ndvi\n(standardized)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """
    #model 1
    #GP regression incidences_district ~ time
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
    fig=pb.subplot(231)
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
    #GP regression incidences_district ~ time + ndvi
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

    fig=pb.subplot(235)
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
    fig=pb.subplot(232)
    mean_,var_,lower_,upper_ = m4.predict(Xlist_[nd])
    GPy.util.plot.gpplot(Xlist_[nd][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd][:,1],Ylist[nd],'kx',mew=1.5)
    pb.plot(Xlist_fut[nd][:,1],Ylist_fut[nd],'rx',mew=1.5)
    if hasattr(m4,'Z'):
        _Z = m4.Z[:m4._M,1]*m4._Zstd[:,1]+m4._Zmean[:,1]
        pb.plot(_Z,np.repeat(pb.ylim()[1]*.1,m4._M),'r|',mew=1.5)
    pb.ylim(ylim)
    pb.ylabel('incidences')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #model 5 plots
    fig=pb.subplot(233)
    mean_,var_,lower_,upper_ = m5.predict(XXlist_[nd])
    GPy.util.plot.gpplot(Xlist_[nd][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd][:,1],Ylist[nd],'kx',mew=1.5)
    pb.plot(Xlist_fut[nd][:,1],Ylist_fut[nd],'rx',mew=1.5)
    if hasattr(m5,'Z'):
        _Z = m5.Z[:m5._M,:]*m5._Zstd+m5._Zmean
        pb.plot(_Z[:,1],np.repeat(pb.ylim()[1]*.1,m5._M),'r|',mew=1.5)
    pb.ylim(ylim)
    pb.ylabel('incidences')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

