"""
Comparison:
    - GP regression incidences_district ~ time
    - GP regression ndvi_district ~ time
    - GP regression incidences_district ~ time + ndvi
    - multioutput GP incidences_distric, ndvi_district ~ time

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

all_stations = ['Arua','Gulu']
R = len(all_stations)
stations = all_stations[:R]
I = np.arange(len(stations))


#Multioutput models outputs/inputs
X1list_ = []
X1list = []
X1list_fut = []

X2list_ = []
X2list = []
X2list_fut = []

Hlist_ = []
Hlist = []
Hlist_fut = []

Y1list = []
Y1list_ = []
Y1list_fut = []

Y2list_ = []
Y2list = []
Y2list_fut = []

YYlist = []
YYlist_ = []
YYlist_fut = []

likelihoods1 = []
likelihoods2 = []
R = 2*len(all_stations)
stations = all_stations
I = np.arange(len(stations))

for i,district in zip(I,stations):
    #data
    #Y1_ = incidences_new[i]
    Y1_ = useful.ndvi_clean(district,'incidences')
    X1_ = useful.ndvi_clean(district,'time')
    aux = useful.raw_data(district,'ndvi')
    Y2_ = aux[:,1][:,None]
    X2_ = aux[:,0][:,None]

    #standradize data incidences and ndvi
    Y1_mean = Y1_.mean()
    Y1_std = Y1_.std()
    Y2_mean = Y2_.mean()
    Y2_std = Y2_.std()

    #Y1_ = (Y1_ - Y1_mean)/Y1_std
    #Y2_ = (Y2_ - Y2_mean)/Y2_std

    #cut
    last = X1_[-1,0]
    cut1 = X1_[X1_ < last - 360].size
    last = X2_[-1,0]
    cut2 = X2_[X2_ < last - 360].size

    Y1 = Y1_[:cut1,:]
    Y1_fut = Y1_[cut1:,:]
    X1 = X1_[:cut1,:]
    X1_fut = X1_[cut1:,:]

    Y2 = Y2_[:cut2,:]
    Y2_fut = Y2_[cut2:,:]
    X2 = X2_[:cut2,:]
    X2_fut = X2_[cut2:,:]

    Y1list_.append(Y1_)
    Y1list.append(Y1)
    Y1list_fut.append(Y1_fut)
    Y2list_.append(Y2_)
    Y2list.append(Y2)
    Y2list_fut.append(Y2_fut)

    likelihoods1.append(GPy.likelihoods.Gaussian(Y1,normalize=False))
    likelihoods2.append(GPy.likelihoods.Gaussian(Y2,normalize=False))

    #Index time
    X1list_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    X1list.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    X1list_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

    X2list_.append(np.hstack([np.repeat(i+len(all_stations),X2_.size)[:,None],X2_]))
    X2list.append(np.hstack([np.repeat(i+len(all_stations),X2.size)[:,None],X2]))
    X2list_fut.append(np.hstack([np.repeat(i+len(all_stations),X2_fut.size)[:,None],X2_fut]))

    #Hlist_.append(np.hstack([np.repeat(i+len(all_stations),X1_.size)[:,None],X1_]))
    #Hlist.append(np.hstack([np.repeat(i+len(all_stations),X1.size)[:,None],X1]))
    #Hlist_fut.append(np.hstack([np.repeat(i+len(all_stations),X1_fut.size)[:,None],X1_fut]))

YYlist_ = Y1list_ + Y2list_
YYlist = Y1list + Y2list
YYlist_fut = Y1list_fut + Y2list_fut

Xlist_ = X1list_ + X2list_
Xlist = X1list + X2list
Xlist_fut = X1list_fut + X2list_fut

likelihoods = likelihoods1 + likelihoods2

#model 6
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

Z = np.linspace(100,1400,20)[:,None]

#m6 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=False)
m6 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m6.ensure_default_constraints()
m6.constrain_positive('kappa')
#m6.tie_param('periodic_*.*_var')
m6.unconstrain('exp_var')
m6.constrain_fixed('exp_var',1)
m6.unconstrain('rbf_rbf_var')
m6.constrain_fixed('rbf_rbf_var',1)
if hasattr(m6,'Z'):
    m6.scale_factor=100#00
    m6.constrain_fixed('iip',m6.Z[:m6._M,1].flatten())
m6.set('exp_len',1.) #=1 if not using log
#m6.unconstrain('icm_rbf_var')
#m6.constrain_fixed('icm_rbf_var',1)
m6.set('icm_rbf_var',5)
m6.set('icm_rbf_len',.0001)
m6.set('W',.01*np.random.rand(R*2))

print m6.checkgrad(verbose=1)
m6.optimize()
print m6

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

    #for nd in range(R):
    #model 6 plots
    fig=pb.subplot(233)
    mean_,var_,lower_,upper_ = m6.predict(Xlist_[nd])
    GPy.util.plot.gpplot(Xlist_[nd][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd][:,1],YYlist[nd],'kx',mew=1.5)
    pb.plot(Xlist_fut[nd][:,1],YYlist_fut[nd],'rx',mew=1.5)
    if hasattr(m6,'Z'):
        #_Z = m6.Z[:m6._M,:]
        #pb.plot(_Z[:,1],np.repeat(pb.ylim()[1]*.1,m6._M),'r|',mew=1.5)
        _Z = m6.Z[:m6._M,1]*m6._Zstd[0,1]+m6._Zmean[0,1]
        pb.plot(_Z,np.repeat(pb.ylim()[1]*.1,m6._M),'r|',mew=1.5)
    #pb.ylim(ylim)
    pb.ylabel('incidences')

    fig=pb.subplot(236)
    mean_,var_,lower_,upper_ = m6.predict(Xlist_[nd+2/2])
    GPy.util.plot.gpplot(Xlist_[nd+2/2][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd+2/2][:,1],YYlist[nd+2/2],'kx',mew=1.5)
    pb.plot(Xlist_fut[nd+2/2][:,1],YYlist_fut[nd+2/2],'rx',mew=1.5)
    if hasattr(m6,'Z'):
        #_Z = m6.Z[:m6._M,:]
        #pb.plot(_Z[:,1],np.repeat(pb.ylim()[1]*.1,m6._M),'r|',mew=1.5)
        _Z = m6.Z[:m6._M,1]*m6._Zstd[0,1]+m6._Zmean[0,1]
        pb.plot(_Z,np.repeat(pb.ylim()[1]*.1,m6._M),'r|',mew=1.5)
    #pb.ylim(ylim)
    pb.ylabel('ndvi')
    #fig.xaxis.set_major_locator(pb.MaxNLocator(6))





