"""
Draft01_20130225_uai examples
---------------------------------

datasets: ../../../playground/malaria/raw_incidence_20130213.dat
datasets: ../../../playground/malaria/raw_ndvi_20130213.dat
datasets: ../../../playground/malaria/raw_met_20130213.dat
datasets: ../../../playground/malaria/raw_geographic_20130213.dat
"""

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
import sys

#My functions
sys.path.append('../../../playground/malaria')
import useful

route = '../../../playground/malaria/drafts/draft01_20130225_uai/'

pb.ion()
pb.close('all')

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
all_stations = malaria_data['stations']
malaria_data.close()

"""
Trends-cycles of incidence and environmental variables
------------------------------------------------------
Arua and Kampala
All variables
"""
"""
examples = ['Arua','Kampala']
var_list_list=[['temperature_min'],['rain'],['humidity_06'],['ndvi']]
name_list = ('temperature','rain','humidity','ndvi')
subplots = np.arange(len(var_list_list)) + 100*len(var_list_list) + 11#(411,412,413,414)
lag_list = (0,0,0,0)
for district in examples:
    longitude,latitude,altitude,area = useful.geo(district)
    incidence,time_i = useful.filtered(district,'incidence',width=2.5,rm_zero=True)
    #incidence,time_i = useful.raw(district,'incidence')
    incidence = (incidence - incidence.mean())/incidence.std()
    pb.figure()
    for var_list,subs,name,lag in zip(var_list_list,subplots,name_list,lag_list):
        for var in var_list:
            fig = pb.subplot(subs)
            weather,time_w = useful.filtered(district,var,width=3,rm_zero=False)
            #weather,time_w = useful.raw(district,var)
            weather = (weather - weather.mean())/weather.std()
            pb.plot(time_i-lag,incidence,'r')
            pb.plot(time_w,weather,'k--',linewidth=1)
            pb.ylabel('%s' %name, size = 15)
            fig.yaxis.set_major_locator(pb.MaxNLocator(3))
            fig.axes.get_xaxis().set_visible(False)
        #pb.text(2,pb.ylim()[1],'lag %s' %lag)
        pb.xlim(0,1800)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)', size = 15)
    pb.suptitle('%s (altitude: %s)' %(district,altitude),size = 18)
    fig_name = '%s_cycle.png' %district
    pb.savefig(route+fig_name)
"""

"""
Multiple outputs vs multiple inputs
-----------------------------------
Gulu
rain
"""
examples = ['Gulu']
var_list = ['rain']
district = examples[0]
var = var_list[0]
cut_date = 1400

#1-output 2-inputs m2
Y_ = useful.weekly_data(district,'incidences')
X1_ = useful.weekly_data(district,'time')
X2_ = useful.weekly_data(district,var)

Y = Y_[X1_ <= cut_date][:,None]
X1 = X1_[X1_ <= cut_date][:,None]
X2 = X2_[X1_ <= cut_date][:,None]

Y_fut = Y_[X1_ > cut_date][:,None]
X1_fut = X1_[X1_ > cut_date][:,None]
X2_fut = X2_[X1_ > cut_date][:,None]
X2_fut = np.repeat(X2.mean(),X1_fut.size)[:,None]

XX_ = np.hstack([X1_,X2_])
XX = np.hstack([X1,X2])
XX_fut = np.hstack([X1_fut,X2_fut])

likelihood2 = GPy.likelihoods.Gaussian(Y,normalize =True)

periodic2 = GPy.kern.periodic_exponential(1)
rbf2 = GPy.kern.rbf(1)
rbf2_ = GPy.kern.rbf(2)
bias2 = GPy.kern.bias(2)
white2 = GPy.kern.white(2)

m2 = GPy.models.GP(XX, likelihood2, GPy.kern.kern.prod_orthogonal(rbf2,periodic2)+rbf2_+white2, normalize_X=True)

m2.tie_param('periodic*.*var')
m2.constrain_positive('var')
m2.constrain_positive('len')
m2.set('exp_len',.1)
m2.set('exp_var',10)
m2.set('rbf_var',.5)
print m2.checkgrad()
m2.optimize()
print m2

fig=pb.subplot(211)
min2_ = X1_.min()
max2_ = X1_.max()
mean_,var_,lower_,upper_ = m2.predict(XX_)
GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
pb.plot(X1,Y,marker='x',color='0.75',mew=1.5)
pb.plot(X1,Y,'kx',mew=1.5)
pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
#pb.ylim(0,upper*1000)
pb.ylabel('incidence')
pb.xlabel('time (days)')

#2-outputs 1-input
additional_outputs_d = [] #Don't include weather-stations data here
stations = district
outputs_s = var_list
outputs_d = ['incidence']
cut_date = 1400

R = 2
I = 1
Ylist_train = []
Xlist_train = []
Ylist_test = []
Xlist_test = []
likelihoods = []

k = 0 #output index
output = 'incidence'
y,x = useful.filtered(district,output,rm_zero=True)
#Train datasets
xtrain = x[x<=cut_date][:,None]
Xlist_train.append( np.hstack([np.repeat(k,xtrain.size)[:,None],xtrain]) )
Ylist_train.append(y[x<=cut_date][:,None])
likelihoods.append(GPy.likelihoods.Gaussian(Ylist_train[-1],normalize=False))
#Test datasets
xtest = x[x>cut_date][:,None]
Xlist_test.append( np.hstack([np.repeat(k,xtest.size)[:,None],xtest]) )
Ylist_test.append(y[x>cut_date][:,None])
#Increase output index
k += 1

output = 'rain'
y,x = useful.filtered(district,output)
#Train datasets
xtrain = x[x<=cut_date][:,None]
Xlist_train.append( np.hstack([np.repeat(k,xtrain.size)[:,None],xtrain]) )
Ylist_train.append(y[x<=cut_date][:,None])
likelihoods.append(GPy.likelihoods.Gaussian(Ylist_train[-1],normalize=False))
#Test datasets
xtest = x[x>cut_date][:,None]
Xlist_test.append( np.hstack([np.repeat(k,xtest.size)[:,None],xtest]) )
Ylist_test.append(y[x>cut_date][:,None])


#Kernel
periodic7 = GPy.kern.periodic_exponential(1)
rbf7 = GPy.kern.rbf(1)
bias7 = GPy.kern.bias(1)
base7_1 = periodic7*rbf7
base_white7 = GPy.kern.white(1)
Dw = 2
kernel7_1 = GPy.kern.icm(base7_1,R,index=0,Dw=Dw)
kernel7_2 = GPy.kern.icm(rbf7.copy(),R,index=0,Dw=Dw)
white7 = GPy.kern.cor_white(base_white7,R,index=0,Dw=Dw)

#Inducing inputs
Z = np.linspace(100,1400,20)[:,None]

#m7 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=False)
m7 = GPy.models.multioutput_GP(Xlist_train, likelihoods, kernel7_1+kernel7_2+white7, Z=Z, normalize_X=True,normalize_Y=True)

m7.ensure_default_constraints()
m7.constrain_positive('kappa')
m7.unconstrain('icm*.*var')
m7.constrain_fixed('icm*.*var',1)
if hasattr(m7,'Z'):
    m7.scale_factor=100
    m7.constrain_fixed('iip',m7.Z[:m7._M,1].flatten())
m7.set('icm_2*.*exp_len',2)
m7.set('icm_2*.*rbf_len',10)
m7.set('icm_1*.*rbf_len',.001)
#m7.set('icm_2*.*exp_len',.1)
#m7.set('icm_2*.*rbf_len',.01)
#m7.set('icm_1*.*rbf_len',.01)
m7.set('W',.01*np.random.rand(2*R*Dw))

print m7.checkgrad(verbose=1)
m7.optimize()
print m7

#incidence - multioutput
fig = pb.subplot(212)
time = np.vstack([ Xlist_train[0],Xlist_test[0] ] )
tmin = time.min()
tmax = time.max()
aux = np.linspace(tmin,tmax,200)[:,None]
index = np.repeat(0,aux.size)[:,None]
X_star = np.hstack([index,aux])
mean_,var_,lower_,upper_ = m7.predict(X_star)
GPy.util.plot.gpplot(X_star[:,1],mean_,lower_,upper_)
pb.plot(Xlist_train[0][:,1],Ylist_train[0],'kx',mew=1.5)
pb.plot(Xlist_test[0][:,1],Ylist_test[0],'rx',mew=1.5)
pb.xlim(0,1800)
pb.ylabel('incidence')
pb.xlabel('time (days)')
fig.xaxis.set_major_locator(pb.MaxNLocator(6))

