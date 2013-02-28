"""
Comparison using samples:
    - GP regression incidences_district ~ time
    - GP regression ndvi_district ~ time
    - GP regression rain_station ~ time
    - GP regression incidences_district ~ time + ndvi
    - multioutput GP incidences_distric, ndvi_district ~ time

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

districts = ['Mbarara','Kasese','Masindi']#,'Wakiso']
additional_outputs_d = ['ndvi'] #Don't include weather-stations data here
stations = ['Mbarara','Kasese','Masindi']#,'Wakiso']
outputs_s = ['rain'] #NOTE this example only supports one output_s

outputs_d = ['incidence'] + additional_outputs_d
cut_date = 1400
ndvi_sample = 100
weather_sample = 150

"""
Sparse multioutput model
"""
print '\nMultioutput model'

R = len(stations)*len(outputs_s) + len(districts)*len(outputs_d)
I = np.arange(len(stations))
Ylist_train = []
Xlist_train = []
Ylist_test = []
Xlist_test = []
likelihoods = []

k = 0 #output index
#Data from districts
for output in outputs_d:
    for district in districts:
        #Geta data
        if output == 'incidence':
            y,x = useful.filtered(district,output,rm_zero=True)
        else:
            y,x = useful.sampled(district,output,size=ndvi_sample)
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

#Data from weather stations
for output in outputs_s:
    for district in stations:
        #Geta data
        y,x = useful.sampled(district,output,size=weather_sample)
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

#Kernel
periodic7 = GPy.kern.periodic_exponential(1)
rbf7 = GPy.kern.rbf(1)
bias7 = GPy.kern.bias(1)
base7 = periodic7*rbf7+rbf7.copy()+bias7
base_white7 = GPy.kern.white(1)
Dw = 2
white7 = GPy.kern.cor_white(base_white7,R,index=0,Dw=Dw)
kernel7 = GPy.kern.icm(base7,R,index=0,Dw=Dw)

#Inducing inputs
Z = np.linspace(100,1400,20)[:,None]

#m7 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=False)
m7 = GPy.models.multioutput_GP(Xlist_train, likelihoods, kernel7+white7, Z=Z, normalize_X=True,normalize_Y=True)

m7.ensure_default_constraints()
m7.constrain_positive('kappa')
#m7.tie_param('periodic_*.*_var')
m7.unconstrain('exp_var')
m7.constrain_fixed('exp_var',1)
m7.unconstrain('rbf_rbf_var')
m7.constrain_fixed('rbf_rbf_var',1)
if hasattr(m7,'Z'):
    m7.scale_factor=100
    m7.constrain_fixed('iip',m7.Z[:m7._M,1].flatten())
m7.set('exp_len',1) #=1 if not using log
m7.set('icm_rbf_var',5)
m7.set('icm_rbf_len',.0001)
m7.set('W',.01*np.random.rand(R*Dw))

print m7.checkgrad(verbose=1)
m7.optimize()
print m7

"""
Incidence regression
"""
Ilist_train = []
Xilist_train = []
Ilist_test = []
Xilist_test = []
likelihoodsi = []
modelsi = []
for district in districts:
    print '\n%s: incidence regression' %district
    #Geta data
    y,x = useful.filtered(district,'incidence',rm_zero=True)
    #Train datasets
    Xilist_train.append(x[x<=cut_date][:,None])
    Ilist_train.append(y[x<=cut_date][:,None])
    likelihoodsi.append(GPy.likelihoods.Gaussian(Ilist_train[-1],normalize=True))
    #Test datasets
    Xilist_test.append(x[x>cut_date][:,None])
    Ilist_test.append(y[x>cut_date][:,None])

    periodici = GPy.kern.periodic_exponential(1)
    rbfi = GPy.kern.rbf(1)
    biasi = GPy.kern.bias(1)
    whitei = GPy.kern.white(1)

    modelsi.append(GPy.models.GP(Xilist_train[-1], likelihoodsi[-1], periodici*rbfi+rbfi.copy()+biasi+whitei, normalize_X=True))

    #modelsi[-1].ensure_default_constraints() #NOTE not working for sum of rbf's
    modelsi[-1].constrain_positive('var')
    modelsi[-1].constrain_positive('len')
    modelsi[-1].tie_param('periodic*.*var')
    print modelsi[-1].checkgrad()
    modelsi[-1].set('exp_len',.001)
    modelsi[-1].set('_rbf_len',.01)
    modelsi[-1].set('exp_var',2)
    modelsi[-1].set('rbf_var',.5)
    modelsi[-1].optimize()
    print modelsi[-1]

"""
ndvi regression
"""
"""
Nlist_train = []
Xnlist_train = []
Nlist_test = []
Xnlist_test = []
likelihoodsn = []
modelsn = []
for output in additional_outputs_d:
    for district in districts:
        print '\n%s: %s regression' %(district,output)
        #Geta data
        y,x = useful.sampled(district,output,size=ndvi_sample)
        #Train datasets
        Xnlist_train.append(x[x<=cut_date][:,None])
        Nlist_train.append(y[x<=cut_date][:,None])
        likelihoodsn.append(GPy.likelihoods.Gaussian(Nlist_train[-1],normalize=True))
        #Test datasets
        Xnlist_test.append(x[x>cut_date][:,None])
        Nlist_test.append(y[x>cut_date][:,None])

        periodicn = GPy.kern.periodic_exponential(1)
        rbfn = GPy.kern.rbf(1)
        biasn = GPy.kern.bias(1)
        whiten = GPy.kern.white(1)

        modelsn.append(GPy.models.GP(Xnlist_train[-1], likelihoodsn[-1], periodicn*rbfn+rbfn.copy()+biasn+whiten, normalize_X=True))

        #modelsn[-1].ensure_default_constraints() #NOTE not working for sum of rbf's
        modelsn[-1].constrain_positive('var')
        modelsn[-1].constrain_positive('len')
        modelsn[-1].tie_param('periodic*.*var')
        modelsn[-1].set('exp_len',.001)
        modelsn[-1].set('_rbf_len',.01)
        modelsn[-1].set('exp_var',2)
        modelsn[-1].set('rbf_var',.5)
        print modelsn[-1].checkgrad()
        modelsn[-1].optimize()
        print modelsn[-1]
"""

"""
Weather outputs regression
"""
"""
Wlist_train = []
Xwlist_train = []
Wlist_test = []
Xwlist_test = []
likelihoodsw = []
modelsw = []
for output in outputs_s:
    for district in stations:
        print '\n%s: %s regression' %(district,output)
        #Geta data
        y,x = useful.sampled(district,output,size=weather_sample)
        #Train datasets
        Xwlist_train.append(x[x<=cut_date][:,None])
        Wlist_train.append(y[x<=cut_date][:,None])
        likelihoodsw.append(GPy.likelihoods.Gaussian(Wlist_train[-1],normalize=True))
        #Test datasets
        Xwlist_test.append(x[x>cut_date][:,None])
        Wlist_test.append(y[x>cut_date][:,None])

        periodicw = GPy.kern.periodic_exponential(1)
        rbfw = GPy.kern.rbf(1)
        biasw = GPy.kern.bias(1)
        whitew = GPy.kern.white(1)

        modelsw.append(GPy.models.GP(Xwlist_train[-1], likelihoodsw[-1], periodicw*rbfw+rbfw.copy()+biasw+whitew, normalize_X=True))

        #modelsw[-1].ensure_default_constraints() #NOTE not working for sum of rbf's
        modelsw[-1].constrain_positive('var')
        modelsw[-1].constrain_positive('len')
        modelsw[-1].tie_param('periodic*.*var')
        modelsw[-1].set('exp_len',.001)
        modelsw[-1].set('_rbf_len',.01)
        modelsw[-1].set('exp_var',2)
        modelsw[-1].set('rbf_var',.5)
        modelsw[-1].optimize()
        print modelsw[-1].checkgrad()
        print modelsw[-1]
"""

"""
Plots
"""
for district,d in zip(districts,range(len(districts))):
    pb.figure()
    pb.suptitle('%s' %district)
    #shifts for multioutput model
    shift = len(districts)

    #incidence - multioutput
    fig = pb.subplot(212)
    time = np.vstack([ Xlist_train[d],Xlist_test[d] ] )
    tmin = time.min()
    tmax = time.max()
    aux = np.linspace(tmin,tmax,200)[:,None]
    index = np.repeat(d,aux.size)[:,None]
    X_star = np.hstack([index,aux])
    mean_,var_,lower_,upper_ = m7.predict(X_star)
    GPy.util.plot.gpplot(X_star[:,1],mean_,lower_,upper_)
    pb.plot(Xlist_train[d][:,1],Ylist_train[d],'kx',mew=1.5)
    pb.plot(Xlist_test[d][:,1],Ylist_test[d],'rx',mew=1.5)
    pb.xlim(0,1800)
    pb.ylabel('incidence')
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #incidence - regression
    fig = pb.subplot(211)
    time = np.vstack([ Xilist_train[d],Xilist_test[d] ] )
    tmin = time.min()
    tmax = time.max()
    X_star = np.linspace(tmin,tmax,200)[:,None]
    mean_,var_,lower_,upper_ = modelsi[d].predict(X_star)
    GPy.util.plot.gpplot(X_star,mean_,lower_,upper_)
    pb.plot(Xilist_train[d],Ilist_train[d],'kx',mew=1.5)
    pb.plot(Xilist_test[d],Ilist_test[d],'rx',mew=1.5)
    pb.xlim(0,1800)
    pb.ylabel('incidence')
    #pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))


    """
    #ndvi - multioutput
    fig = pb.subplot(235)
    time = np.vstack([ Xlist_train[d+shift],Xlist_test[d+shift] ] )
    tmin = time.min()
    tmax = time.max()
    aux = np.linspace(tmin,tmax,200)[:,None]
    index = np.repeat(d+shift,aux.size)[:,None]
    X_star = np.hstack([index,aux])
    mean_,var_,lower_,upper_ = m7.predict(X_star)
    GPy.util.plot.gpplot(X_star[:,1],mean_,lower_,upper_)
    pb.plot(Xlist_train[d+shift][:,1],Ylist_train[d+shift],'kx',mew=1.5)
    pb.plot(Xlist_test[d+shift][:,1],Ylist_test[d+shift],'rx',mew=1.5)
    pb.xlim(0,1800)
    pb.ylabel(additional_outputs_d[0])
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #weather - multioutput
    fig = pb.subplot(234)
    time = np.vstack([ Xlist_train[d+2*shift],Xlist_test[d+2*shift] ] )
    tmin = time.min()
    tmax = time.max()
    aux = np.linspace(tmin,tmax,200)[:,None]
    index = np.repeat(d+2*shift,aux.size)[:,None]
    X_star = np.hstack([index,aux])
    mean_,var_,lower_,upper_ = m7.predict(X_star)
    GPy.util.plot.gpplot(X_star[:,1],mean_,lower_,upper_)
    pb.plot(Xlist_train[d+2*shift][:,1],Ylist_train[d+2*shift],'kx',mew=1.5)
    pb.plot(Xlist_test[d+2*shift][:,1],Ylist_test[d+2*shift],'rx',mew=1.5)
    pb.xlim(0,1800)
    pb.ylabel(outputs_d[0])
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """

    """
    #ndvi - regression
    fig = pb.subplot(232)
    time = np.vstack([ Xnlist_train[d],Xnlist_test[d] ] )
    tmin = time.min()
    tmax = time.max()
    X_star = np.linspace(tmin,tmax,200)[:,None]
    mean_,var_,lower_,upper_ = modelsn[d].predict(X_star)
    GPy.util.plot.gpplot(X_star,mean_,lower_,upper_)
    pb.plot(Xnlist_train[d],Nlist_train[d],'kx',mew=1.5)
    pb.plot(Xnlist_test[d],Nlist_test[d],'rx',mew=1.5)
    pb.xlim(0,1800)
    pb.ylabel('ndvi')
    #pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #weather - regression
    fig = pb.subplot(231)
    time = np.vstack([ Xwlist_train[d],Xwlist_test[d] ] )
    tmin = time.min()
    tmax = time.max()
    X_star = np.linspace(tmin,tmax,200)[:,None]
    mean_,var_,lower_,upper_ = modelsw[d].predict(X_star)
    GPy.util.plot.gpplot(X_star,mean_,lower_,upper_)
    pb.plot(Xwlist_train[d],Wlist_train[d],'kx',mew=1.5)
    pb.plot(Xwlist_test[d],Wlist_test[d],'rx',mew=1.5)
    pb.ylabel(outputs_s[0])
    pb.xlim(0,1800)
    #pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """

