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
ndvi
"""
examples = ['Soroti'] #Soroti Masindi Mbarara
var_list = ['ndvi']
district = examples[0]
var = var_list[0]
cut_date = 1400

#1-output 2-inputs m2
Y_0 = useful.weekly_data(district,'incidences')
X1_0 = useful.weekly_data(district,'time')
X2_0 = useful.weekly_data(district,var)
#Remove outliers
Y_ = Y_0[np.logical_and(Y_0>831,Y_0<3388)][:,None]
X1_ = X1_0[np.logical_and(Y_0>831,Y_0<3388)][:,None]
X2_ = X2_0[np.logical_and(Y_0>831,Y_0<3388)][:,None]
#Y_ = Y_0
#X1_ = X1_0
#X2_ = X2_0


Y = Y_[X1_ <= cut_date][:,None]
X1 = X1_[X1_ <= cut_date][:,None]
X2 = X2_[X1_ <= cut_date][:,None]
Y_fut = Y_[X1_ > cut_date][:,None]
X1_fut = X1_[X1_ > cut_date][:,None]
X2_fut = X2_[X1_ > cut_date][:,None]
#X2_fut = np.repeat(X2.mean(),X1_fut.size)[:,None]
XX_ = np.hstack([X1_,X2_])
XX = np.hstack([X1,X2])
#XX_fut = np.hstack([X1_fut,X2_fut])
X2_mean = np.repeat(X2.mean(),200)[:,None]
X1_pred = np.linspace(0,1800,200)[:,None]
XX_pred = np.hstack([X1_pred,X2_mean])

likelihood2 = GPy.likelihoods.Gaussian(Y_,normalize =True)

rbf2_1 = GPy.kern.rbf(2,lengthscale=.10,variance=.1)
rbf2_2 = GPy.kern.rbf(2,lengthscale=3.60,variance=2)
bias2 = GPy.kern.bias(2)
white2 = GPy.kern.white(2)

m2 = GPy.models.GP(XX_, likelihood2, rbf2_1+rbf2_2+white2, normalize_X=True)
m2.constrain_positive('var')
m2.constrain_positive('len')
print m2.checkgrad()
m2.optimize()
print m2

min2_ = X1_.min()
max2_ = X1_.max()
mean_,var_,lower_,upper_ = m2.predict(XX_pred)
fig=pb.subplot(311)
GPy.util.plot.gpplot(X1_pred,mean_,lower_,upper_)
pb.ylabel('GP regression\nincidence',size=15)
#pb.xlabel('time (days)',size=15)
LY,UY = pb.ylim()
UY = 1.2*UY

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
#xtrain = x[x<=cut_date][:,None]
xtrain = x
Xlist_train.append( np.hstack([np.repeat(k,xtrain.size)[:,None],xtrain]) )
#Ylist_train.append(y[x<=cut_date][:,None])
Ylist_train.append(y)
likelihoods.append(GPy.likelihoods.Gaussian(Ylist_train[-1],normalize=False))
#Test datasets
xtest = x[x>cut_date][:,None]
Xlist_test.append( np.hstack([np.repeat(k,xtest.size)[:,None],xtest]) )
Ylist_test.append(y[x>cut_date][:,None])
#Increase output index
k += 1

output = 'ndvi'
y,x = useful.filtered(district,output,width=2)
#Train datasets
#xtrain = x[x<=cut_date][:,None]
xtrain = x
Xlist_train.append( np.hstack([np.repeat(k,xtrain.size)[:,None],xtrain]) )
#Ylist_train.append(y[x<=cut_date][:,None])
Ylist_train.append(y)
likelihoods.append(GPy.likelihoods.Gaussian(Ylist_train[-1],normalize=False))
#Test datasets
xtest = x[x>cut_date][:,None]
xtest = x[x>cut_date][:,None]
Xlist_test.append( np.hstack([np.repeat(k,xtest.size)[:,None],xtest]) )
Ylist_test.append(y[x>cut_date][:,None])

periodic7 = GPy.kern.periodic_exponential(1)
rbf7_1 = GPy.kern.rbf(1,lengthscale=30)
rbf7_2 = GPy.kern.rbf(1,lengthscale=360)
bias7 = GPy.kern.bias(1)
base_white7 = GPy.kern.white(1)
Dw = 2
kernel7_1 = GPy.kern.icm(rbf7_1,R,index=0,Dw=Dw)
kernel7_2 = GPy.kern.icm(rbf7_2,R,index=0,Dw=Dw)
white7 = GPy.kern.cor_white(base_white7,R,index=0,Dw=Dw)
#Inducing inputs
Z = np.linspace(100,1800,20)[:,None]

#m7 = GPy.models.mGP(Xlist_train, likelihoods, kernel7_1+kernel7_2+white7, normalize_X=True,normalize_Y=True)
m7 = GPy.models.multioutput_GP(Xlist_train, likelihoods, kernel7_1+kernel7_2+white7, Z=Z, normalize_X=True,normalize_Y=True)
m7.ensure_default_constraints()
m7.constrain_positive('kappa')
m7.unconstrain('icm*.*var')
m7.constrain_fixed('icm*.*var',1)
m7.scale_factor=100
m7.constrain_fixed('iip',m7.Z[:m7._M,1].flatten())
#m7.set('icm_2*.*exp_len',2)
#m7.set('icm_2*.*rbf_len',.1)
#m7.set('icm_1*.*rbf_len',.01)
m7.set('1_W',.01*np.random.rand(R*Dw))
m7.set('2_W',.01*np.random.rand(R*Dw))

print m7.checkgrad(verbose=1)
m7.optimize()
print m7

#Plots

#incidence - multioutput
#time = np.vstack([ Xlist_train[0],Xlist_test[0] ] )
time =Xlist_train[0]
tmin = 0#time.min()
tmax = 1800#time.max()
aux = np.linspace(tmin,tmax,200)[:,None]
index = np.repeat(0,aux.size)[:,None]
indexw = np.repeat(1,aux.size)[:,None]
X_star = np.hstack([index,aux])
Xw_star = np.hstack([indexw,aux])
mean_,var_,lower_,upper_ = m7.predict(X_star)
LY = min(LY,mean_.min())
UY = max(UY,mean_.min())*1.1
#pb.ylim(LY,UY)

#Previous plot
#pb.plot(Xlist_train[0][:,1],Ylist_train[0],'bx',mew=1.5)
#pb.plot(Xlist_test[0][:,1],Ylist_test[0],'bx',mew=1.5)
#pb.plot(Xlist_train[0][:,1],Ylist_train[0],'rx',mew=1.5)
pb.plot(X1_,Y_,'kx',mew=1.5)
#pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
pb.xlim(0,1800)
pb.ylim(0,UY)
fig.xaxis.set_major_locator(pb.MaxNLocator(5))
#incidence
fig = pb.subplot(312)
GPy.util.plot.gpplot(X_star[:,1],mean_,lower_,upper_)
pb.plot(Xlist_train[0][:,1],Ylist_train[0],'kx',mew=1.5)
#pb.plot(Xlist_test[0][:,1],Ylist_test[0],'rx',mew=1.5)

_Z = m7.Z[:m7._M,1]*m7._Zstd[0,1]+m7._Zmean[0,1]
pb.plot(_Z,np.repeat(500,m7._M),'r|',mew=1.5)
pb.ylabel('multiple output\nincidence',size=15)
#pb.xlabel('time (days)',size=15)
fig.xaxis.set_major_locator(pb.MaxNLocator(5))
pb.suptitle('%s' %district,size = 18)
#pb.ylim(LY,UY)
pb.ylim(0,UY)
pb.xlim(0,1800)
fig_name = '%s_example01.png' %district
pb.savefig(route+fig_name)

#weather
fig = pb.subplot(313)
mean_w,var_w,lower_w,upper_w = m7.predict(Xw_star)
GPy.util.plot.gpplot(Xw_star[:,1],mean_w,lower_w,upper_w)
pb.plot(Xlist_train[1][:,1],Ylist_train[1],'kx',mew=1.5)
#pb.plot(Xlist_test[1][:,1],Ylist_test[1],'rx',mew=1.5)

_Z = m7.Z[:m7._M,1]*m7._Zstd[0,1]+m7._Zmean[0,1]
pb.plot(_Z,np.repeat(pb.ylim()[0],m7._M),'r|',mew=1.5)
pb.xlim(0,1800)
pb.ylabel('multiple output\nndvi',size=15)
pb.xlabel('time (days)',size=15)
fig.xaxis.set_major_locator(pb.MaxNLocator(5))
#pb.suptitle('%s' %district,size = 18)
fig_name = '%s_example01.png' %district
pb.savefig(route+fig_name)
