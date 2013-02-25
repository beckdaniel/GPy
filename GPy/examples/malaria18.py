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

#Multioutput models outputs/inputs
Xlist_ = []
Xlist = []
Xlist_fut = []

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
    Y1_ = useful.ndvi_clean(district,'incidences')
    Y2_ = useful.ndvi_clean(district,'ndvi')
    X1_ = useful.ndvi_clean(district,'time')

    #standradize data incidences and ndvi
    Y1_mean = Y1_.mean()
    Y1_std = Y1_.std()
    Y2_mean = Y2_.mean()
    Y2_std = Y2_.std()

    #cut
    last = X1_[-1,0]
    cut = X1_[X1_ < last - 360].size

    Y1 = Y1_[:cut,:]
    Y1_fut = Y1_[cut:,:]

    Y2 = Y2_[:cut,:]
    Y2_fut = Y2_[cut:,:]

    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]

    #XX_ = np.hstack([X1_,X2_])
    #XX = np.hstack([X1,X2])
    #XX_fut = np.hstack([X1_fut,X2_fut])

    #likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    Y1list_.append(Y1_)
    Y1list.append(Y1)
    Y1list_fut.append(Y1_fut)

    Y2list_.append(Y2_)
    Y2list.append(Y2)
    Y2list_fut.append(Y2_fut)

    likelihoods1.append(GPy.likelihoods.Gaussian(Y1,normalize=False))
    likelihoods2.append(GPy.likelihoods.Gaussian(Y2,normalize=False))

    #Index time
    Xlist_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    Xlist.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    Xlist_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))
    #Xlist_.append(X1_)
    #Xlist.append(X1)
    #Xlist_fut.append(X1_fut)
    Hlist_.append(np.hstack([np.repeat(i+len(all_stations),X1_.size)[:,None],X1_]))
    Hlist.append(np.hstack([np.repeat(i+len(all_stations),X1.size)[:,None],X1]))
    Hlist_fut.append(np.hstack([np.repeat(i+len(all_stations),X1_fut.size)[:,None],X1_fut]))

YYlist_ = Y1list_ + Y2list_
YYlist = Y1list + Y2list
YYlist_fut = Y1list_fut + Y2list_fut

Xlist_ = Xlist_ + Hlist_
Xlist = Xlist + Hlist
Xlist_fut = Xlist_fut + Hlist_fut

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

Z = np.linspace(100,1400,5)[:,None]

#m6 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=True)
m6 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m6.ensure_default_constraints()
m6.constrain_positive('kappa')
m6.unconstrain('exp_var')
m6.constrain_fixed('exp_var',1)
if hasattr(m6,'Z'):
    m6.scale_factor=100#00
    m6.constrain_fixed('iip',m6.Z[:m6._M,1].flatten())
m6.set('exp_len',.1)
m6.set('icm_rbf_var',10)
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

