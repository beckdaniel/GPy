"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/uganda_data_20130213.dat
B matrix controls the relation between districts
Incidences are assumed to have a log-normal distribution
"""
#NOTE This is a non-sparse model

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
pb.ion()
pb.close('all')

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
all_stations = malaria_data['stations']
station_variables = malaria_data['headers_daily']

#Define districts to analyze
d_names = ['Arua']
d_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == d_i] for d_i in d_names])
if len(d_names) > len(d_numbers):
    print 'Warning: some districts were not found in malaria_data'

#Define outputs from non-weather data
Y_names_1 = ['incidences']
Y_numbers_1 = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == Y_i] for Y_i in Y_names_1])
Y_list_1 = [malaria_data['data'][d_i][:,Y_numbers_1] for d_i in d_numbers]
if len(Y_names_1) > 1:
    for num_i in range(len(d_numbers)):
        Y_list_1[num_i] = np.vstack(Y_list_1[num_i])

#Define stations to analyze
s_names = ['Arua']
s_numbers = np.hstack([np.arange(len(malaria_data['stations']))[np.array(malaria_data['stations']) == s_i] for s_i in s_names])
if len(s_names) > len(s_numbers):
    print 'Warning: some stations were not found in malaria_data'

#Define outputs from weather data
Y_names_2 = ['humidity_12']#,'humidity_06']#,'humidity_12','temperature_min','temperature_max']
Y_numbers_2 = np.hstack([np.arange(len(malaria_data['headers_daily']))[np.array(malaria_data['headers_daily']) == Y_i] for Y_i in Y_names_2])
Y_list_2 = []
for sn in s_numbers:
    for yn in Y_numbers_2:
        Y_list_2.append(malaria_data['weather_daily'][sn][:,yn][:,None])

#Define input from non-weather data
X_names_1 = ['district','time','rain']#,'time']#,'rain','ndvi','humidity_06','humidity_12','rain','temperature_min','temperature_max']
X_numbers_1 = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == n_i] for n_i in X_names_1])
X_list_1 = [malaria_data['data'][d_i][:,X_numbers_1] for d_i in d_numbers]
for X_i,num_i in zip(X_list_1,range(len(d_names))):
    X_i[:,0] = np.repeat(num_i,X_i.shape[0]) #Change district number according to data analyzed

#Define input from weather data
X_names_2 = ['rain','time']#,'time']#,'rain','ndvi','humidity_06','humidity_12','rain','temperature_min','temperature_max']
X_numbers_2 = np.hstack([np.arange(len(malaria_data['headers_daily']))[np.array(malaria_data['headers_daily']) == n_i] for n_i in X_names_2])
X_list_2 = []
for sn in s_numbers:
    for yn in Y_numbers_2:
        X_list_2.append(malaria_data['weather_daily'][sn][:,X_numbers_2])
for X_i,num_i in zip(X_list_2,range(len(Y_names_2))):
    X_i[:,0] = np.repeat(1+num_i,X_i.shape[0]) #Change district number according to data analyzed

#Mix lists
X_list = []
Y_list = []
for j in range(len(Y_list_1)):
    Y_list.append(Y_list_1[j])
    X_list.append(X_list_1[j])
for j in range(len(Y_list_2)):
    Y_list.append(Y_list_2[j])
    X_list.append(X_list_2[j])

#Close data file
malaria_data.close()

#Create likelihood
likelihoods = []
for Y_i in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(Y_i,normalize=True))

#Define coreg_kern and base kernels
R = len(X_list)
D = X_list[0].shape[1] - 1
rbf = GPy.kern.rbf(D)
noise = GPy.kern.white(D)
base = rbf + rbf.copy() + noise
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

Y = np.log(Y_list_1[0][1:]/Y_list_1[0][:-1])
X = X_list_1[0][1:,1]

Y2 = np.log(X_list_1[0][1:,2]/X_list_1[0][:-1,2])
#X2 = X_list_2[0][1:,1]

Yraw = Y_list_1[0][1:]

pb.subplot(411)
pb.plot(X[:-1],Y[:-1],'r')
pb.xlim(0,2000)
pb.subplot(412)
pb.plot(X[1:],Y2[:-1],'b') #+30
pb.xlim(0,2000)
pb.subplot(413)
pb.plot(X[1:],X_list_1[0][1:-1,2],'b')
pb.xlim(0,2000)
pb.subplot(414)
pb.plot(X,(Yraw - Yraw.mean())/Yraw.std(),'r')
pb.xlim(0,2000)

rbf1 = GPy.kern.rbf(2)
rbf2 = GPy.kern.rbf(2)
noise = GPy.kern.white(2)
kernel = rbf1 + rbf2 + noise
likelihood=GPy.likelihoods.Gaussian(Y[:-1],normalize=False)
X = np.hstack([X[:-1][:,None],Y2[:-1][:,None]])

m = GPy.models.GP(X, likelihood, kernel, normalize_X=True)
#Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.set('_1_len',10)
m.set('_2_len',.1)
#Optimize
print m.checkgrad(verbose=True)
m.optimize()
pb.figure()
m.plot()

rbf1_ = GPy.kern.rbf(1)
rbf2_ = GPy.kern.rbf(1)
noise_ = GPy.kern.white(1)
kernel_ = rbf1_ + rbf2_
likelihood_ = GPy.likelihoods.Gaussian(Y[:-1],normalize=False)
#X_ = Y2[:-1][:,None]
X_ = X[:,0][:,None]

q = GPy.models.GP(X_, likelihood_, kernel_, normalize_X=True)
#Constraints
q.scale_factor = 1.
q.ensure_default_constraints()
q.set('_1_len',10)
q.set('_2_len',.1)
#Optimize
print q.checkgrad(verbose=True)
q.optimize()
pb.figure()
q.plot()
print q
