"""
Sparse/Full Multioutput GP for malaria counts

outputs: incidences_district
inputs: time

---------------------------------

dataset: ../../../playground/malaria/uganda_ndvi_20130213.dat

B matrix controls the relation between districts
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

#Forcast
Xlist_ = []
Xlist = []
Xlist_fut = []
Ylist = []
Ylist_ = []
Ylist_fut = []
likelihoods = []
R = 4
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

    likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    Ylist_.append(Y_)
    Ylist.append(Y)
    Ylist_fut.append(Y_fut)

    #Index time
    Xlist_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    Xlist.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    Xlist_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

#model 4
print '\nmodel 4'

R = R
D = 1
periodic4 = GPy.kern.periodic_exponential(1)
rbf4 = GPy.kern.rbf(1)
linear4 = GPy.kern.linear(1)
bias4 = GPy.kern.bias(1)
base_white4 = GPy.kern.white(1)
white4 = GPy.kern.cor_white(base_white4,R,index=0,Dw=2) #FIXME
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
