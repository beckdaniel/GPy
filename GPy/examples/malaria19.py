"""
Sparse multioutput model:
    - outputs: incidences_district and ndvi_district
    - inputs: time
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

all_stations = ['Mubende','Masindi']#,'Mbarara','Kampala','Kasese']

#Clean incidence series
incidences_new = []
for district in all_stations:
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

    pb.figure()
    #fig=pb.subplot(231)
    min1_ = X1_.min()
    max1_ = X1_.max()
    mean_,var_,lower_,upper_ = m1.predict(X1_)
    incidences_new.append(mean_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    #ylim=pb.ylim()
    #fig.xaxis.set_major_locator(pb.MaxNLocator(6))


#Multioutput models outputs/inputs
X1list_ = []
X1list = []
X1list_fut = []

X2list_ = []
X2list = []
X2list_fut = []

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
R = len(all_stations)
stations = all_stations
I = np.arange(len(stations))

for i,district in zip(I,stations):
    #data
    #Y1_ = useful.ndvi_clean(district,'incidences')
    Y1_ = incidences_new[i]
    X1_ = useful.ndvi_clean(district,'time')
    #aux = useful.raw_data(district,'ndvi')
    #Y2_ = aux[:,1][:,None]
    #X2_ = aux[:,0][:,None]

    #standradize data incidences and ndvi
    #Y1_mean = Y1_.mean()
    #Y1_std = Y1_.std()
    #Y2_mean = Y2_.mean()
    #Y2_std = Y2_.std()

    #Y1_ = (Y1_ - Y1_mean)/Y1_std
    #Y2_ = (Y2_ - Y2_mean)/Y2_std

    #cut
    last = X1_[-1,0]
    cut1 = X1_[X1_ < last - 360].size
    #last = X2_[-1,0]
    #cut2 = X2_[X2_ < last - 360].size

    Y1 = Y1_[:cut1,:]
    Y1_fut = Y1_[cut1:,:]
    X1 = X1_[:cut1,:]
    X1_fut = X1_[cut1:,:]

    #Y2 = Y2_[:cut2,:]
    #Y2_fut = Y2_[cut2:,:]
    #X2 = X2_[:cut2,:]
    #X2_fut = X2_[cut2:,:]

    Y1list_.append(Y1_)
    Y1list.append(Y1)
    Y1list_fut.append(Y1_fut)
    #Y2list_.append(Y2_)
    #Y2list.append(Y2)
    #Y2list_fut.append(Y2_fut)

    likelihoods1.append(GPy.likelihoods.Gaussian(Y1,normalize=False))
    #likelihoods2.append(GPy.likelihoods.Gaussian(Y2,normalize=False))

    #Index time
    X1list_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    X1list.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    X1list_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

    #X2list_.append(np.hstack([np.repeat(i+len(all_stations),X2_.size)[:,None],X2_]))
    #X2list.append(np.hstack([np.repeat(i+len(all_stations),X2.size)[:,None],X2]))
    #X2list_fut.append(np.hstack([np.repeat(i+len(all_stations),X2_fut.size)[:,None],X2_fut]))


YYlist_ = Y1list_# + Y2list_
YYlist = Y1list# + Y2list
YYlist_fut = Y1list_fut# + Y2list_fut

Xlist_ = X1list_# + X2list_
Xlist = X1list# + X2list
Xlist_fut = X1list_fut# + X2list_fut

likelihoods = likelihoods1# + likelihoods2

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
base4 = periodic4*rbf4+rbf4.copy()#+bias4
kernel4 = GPy.kern.icm(base4,R,index=0,Dw=2)

Z = np.linspace(100,1400,9)[:,None]

#m6 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=True)
m6 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m6.ensure_default_constraints()
m6.constrain_positive('kappa')
m6.unconstrain('exp_var')
m6.constrain_fixed('exp_var',1)
if hasattr(m6,'Z'):
    m6.scale_factor=100#00
    m6.constrain_fixed('iip',m6.Z[:m6._M,1].flatten())
m6.set('exp_len',1.)
m6.set('icm_rbf_var',2)
m6.set('W',.001*np.random.rand(R*2))

print m6.checkgrad(verbose=1)
m6.optimize()
print m6

for nd in range(R):
    #model 6 plots
    #fig=pb.subplot(233)
    fig = pb.figure()
    mean_,var_,lower_,upper_ = m6.predict(Xlist_[nd])
    GPy.util.plot.gpplot(Xlist_[nd][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd][:,1],YYlist[nd],'kx',mew=1.5)
    print '%s' %nd
    print Xlist[nd][:,1].size,YYlist[nd].size
    pb.plot(Xlist_fut[nd][:,1],YYlist_fut[nd],'rx',mew=1.5)
    if hasattr(m6,'Z'):
        #_Z = m6.Z[:m6._M,:]*m6._Zstd+m6._Zmean
        _Z = m6.Z[:m6._M,:]
        pb.plot(_Z[:,1],np.repeat(pb.ylim()[1]*.1,m6._M),'r|',mew=1.5)
    #pb.ylim(ylim)
    if nd < R/2:
        pb.ylabel('incidences')
    else:
        pb.ylabel('ndvi')
    #fig.xaxis.set_major_locator(pb.MaxNLocator(6))
