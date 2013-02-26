"""
Sparse multioutput model:
    - outputs: incidences_district, rain_station, and ndvi_district
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

all_stations = ['Arua']#,'Gulu']#,'Mbarara','Kampala','Kasese']

#Multioutput models outputs/inputs
X1list_ = []
X1list = []
X1list_fut = []

X2list_ = []
X2list = []
X2list_fut = []

X3list_ = []
X3list = []
X3list_fut = []

Y1list = []
Y1list_ = []
Y1list_fut = []

Y2list_ = []
Y2list = []
Y2list_fut = []

Y3list_ = []
Y3list = []
Y3list_fut = []

YYlist = []
YYlist_ = []
YYlist_fut = []

likelihoods1 = []
likelihoods2 = []
likelihoods3 = []
R = 3*len(all_stations)
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
    aux = useful.raw_data(district,'rain')
    Y3_ = aux[:,1][:,None]
    X3_ = aux[:,0][:,None] - 7 #NOTE 7 days of lag considered

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
    last = X2_[-1,0]
    cut2 = X2_[X2_ < last - 360].size
    last = X3_[-1,0]
    cut3 = X3_[X3_ < last - 360].size

    Y1 = Y1_[:cut1,:]
    Y1_fut = Y1_[cut1:,:]
    X1 = X1_[:cut1,:]
    X1_fut = X1_[cut1:,:]

    Y2 = Y2_[:cut2,:]
    Y2_fut = Y2_[cut2:,:]
    X2 = X2_[:cut2,:]
    X2_fut = X2_[cut2:,:]

    Y3 = Y3_[:cut3,:]
    Y3_fut = Y3_[cut3:,:]
    X3 = X3_[:cut3,:]
    X3_fut = X3_[cut3:,:]

    Y1list_.append(Y1_)
    Y1list.append(Y1)
    Y1list_fut.append(Y1_fut)
    Y2list_.append(Y2_)
    Y2list.append(Y2)
    Y2list_fut.append(Y2_fut)
    Y3list_.append(Y3_)
    Y3list.append(Y3)
    Y3list_fut.append(Y3_fut)

    likelihoods1.append(GPy.likelihoods.Gaussian(Y1,normalize=False))
    likelihoods2.append(GPy.likelihoods.Gaussian(Y2,normalize=False))
    likelihoods3.append(GPy.likelihoods.Gaussian(Y3,normalize=False))

    #Index time
    X1list_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    X1list.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    X1list_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

    X2list_.append(np.hstack([np.repeat(i+len(all_stations),X2_.size)[:,None],X2_]))
    X2list.append(np.hstack([np.repeat(i+len(all_stations),X2.size)[:,None],X2]))
    X2list_fut.append(np.hstack([np.repeat(i+len(all_stations),X2_fut.size)[:,None],X2_fut]))

    X3list_.append(np.hstack([np.repeat(i+len(all_stations),X3_.size)[:,None],X3_]))
    X3list.append(np.hstack([np.repeat(i+len(all_stations),X3.size)[:,None],X3]))
    X3list_fut.append(np.hstack([np.repeat(i+len(all_stations),X3_fut.size)[:,None],X3_fut]))

YYlist_ = Y1list_ + Y2list_ + Y3list_
YYlist = Y1list + Y2list + Y3list
YYlist_fut = Y1list_fut + Y2list_fut + Y3list_fut

Xlist_ = X1list_ + X2list_ + X3list_
Xlist = X1list + X2list + X3list
Xlist_fut = X1list_fut + X2list_fut + X3list_fut

likelihoods = likelihoods1 + likelihoods2 + likelihoods3

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

Z = np.linspace(100,1400,9)[:,None]

#m6 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=False)
m6 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m6.ensure_default_constraints()
m6.constrain_positive('kappa')
m6.unconstrain('exp_var')
m6.constrain_fixed('exp_var',1)
if hasattr(m6,'Z'):
    m6.scale_factor=100#00
    m6.constrain_fixed('iip',m6.Z[:m6._M,1].flatten())
m6.set('exp_len',1) #=1 if not using log
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
    if nd < R/3:
        pb.ylabel('incidences')
    elif nd < 2*R/3:
        pb.ylabel('ndvi')
    else:
        pb.ylabel('rain')
    #fig.xaxis.set_major_locator(pb.MaxNLocator(6))
