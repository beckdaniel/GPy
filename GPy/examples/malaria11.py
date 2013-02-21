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
d_names = ['Soroti']
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
s_names = ['Soroti']
s_numbers = np.hstack([np.arange(len(malaria_data['stations']))[np.array(malaria_data['stations']) == s_i] for s_i in s_names])
if len(s_names) > len(s_numbers):
    print 'Warning: some stations were not found in malaria_data'

#Define outputs from weather data
Y_names_2 = ['humidity_06']#,'humidity_06']#,'humidity_12','temperature_min','temperature_max']
Y_numbers_2 = np.hstack([np.arange(len(malaria_data['headers_daily']))[np.array(malaria_data['headers_daily']) == Y_i] for Y_i in Y_names_2])
Y_list_2 = []
for sn in s_numbers:
    for yn in Y_numbers_2:
        Y_list_2.append(malaria_data['weather_daily'][sn][:,yn][:,None])

#Define input from non-weather data
X_names_1 = ['district','time']#,'time']#,'rain','ndvi','humidity_06','humidity_12','rain','temperature_min','temperature_max']
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

#Define model
m = GPy.models.mGP(X_list, likelihoods, kernel, normalize_X=True,normalize_Y=False)

#Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_1_var')
m.constrain_fixed('rbf_1_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.unconstrain('rbf_2_var')
m.constrain_fixed('rbf_2_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
m.set('_1_len',10)
m.set('_2_len',.1)
m.set('W',np.random.rand(R*2))
#Optimize
print m.checkgrad(verbose=True)
m.optimize()

#Plots 1D or 2D
o_names = ['incidence','rain']
for os,dn in zip(m.Xos,o_names):
    pb.figure()
    pb.subplot(211)
    m.plot_f(which_data = os)
    pb.title('%s' %dn)

    pb.subplot(212)
    m.plot(which_data = os)

#for os,on in zip(m.Xos,range(len(m.Xos))):
#    for vn in range(len(X_names)-1):
#        pb.figure()
#        m.plot_HD(which_input=vn,which_data=os)
#        pb.title('%s: %s' %(d_names[on],X_names[vn+1]))

#Print model
print m

#Print B matrix
print np.round(m.kern.parts[0].B,2)
"""
#Predict districts
malaria_data = shelve.open('../../../playground/malaria/malaria_data_20130213.dat',writeback=False)

p_names = ['Mubende']
p_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == p_i] for p_i in p_names])
if len(p_names) > len(p_numbers):
    print 'Warning: some districts were not found in malaria_data'

#Define Xnew values
#X_names = same X_names
#X_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == n_i] for n_i in X_names])
Xnew_list = [malaria_data['data'][p_i][:,X_numbers] for p_i in p_numbers]
for X_i,num_i in zip(Xnew_list,range(len(p_names))):
    X_i[:,0] = np.repeat(num_i,X_i.shape[0]) #Change district number according to data analyzed

malaria_data.close()
"""
