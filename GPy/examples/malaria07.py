"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/malaria_data20130213.dat
B matrix controls the relation between incidence and rain
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
malaria_data = shelve.open('../../../playground/malaria/malaria_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']

#Define districts to analyze
d_names = ['Mubende','Nakasongola','Kamuli','Kampala','Mukono','Luwero','Tororo']
d_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == d_i] for d_i in d_names])

#Define output
Y_names = ['incidences','rain']
Y_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == Y_i] for Y_i in Y_names])
Y_list = []
for output_i in Y_numbers:
    Y_list.append([malaria_data['data'][d_i][:,output_i] for d_i in d_numbers])
    Y_list[-1] = np.hstack(Y_list[-1])[:,None]

#Define input
X_names = ['district','time','ndvi','longitude','latitude','altitude']
X_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == n_i] for n_i in X_names])
X_list = []
for output_i,new_num_i in zip(Y_numbers,range(len(Y_numbers))):
    X_list.append([malaria_data['data'][d_i][:,X_numbers] for d_i in d_numbers])
    X_list[-1] = np.vstack(X_list[-1])
    X_list[-1][:,0] = np.repeat(new_num_i,X_list[-1][:,0].size)

#Close data file
malaria_data.close()

#Create likelihood
likelihoods = []
for Y_i in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(Y_i,normalize=True))

#Define coreg_kern and base kernels
R = len(Y_names)
D = len(X_names) - 1 #-1 if district is in X_names
rbf = GPy.kern.rbf(D)
noise = GPy.kern.white(D)
base = rbf + rbf.copy() + noise
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

#Define model
m = GPy.models.mGP(X_list, likelihoods, kernel, normalize_X=True,normalize_Y=False)

# Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_1_var')
m.constrain_fixed('rbf_1_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.unconstrain('rbf_2_var')
m.constrain_fixed('rbf_2_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
m.set('_1_len',.10)
m.set('_2_len',.1)
m.set('W',np.random.rand(R*2))

#Optimize
print m.checkgrad(verbose=True)
m.optimize()
"""
#Plots 1D or 2D
for os,dn in zip(m.Xos,d_names):
    pb.figure()
    pb.subplot(211)
    m.plot_f(which_data = os)
    pb.title('%s' %dn)

    pb.subplot(212)
    m.plot(which_data = os)
"""
for os,on in zip(m.Xos,range(len(m.Xos))):
    for vn in range(len(X_names)-1):
        pb.figure()
        m.plot_HD(which_input=vn,which_data=os)
        pb.title('%s vs %s' %(Y_names[on],X_names[vn+1]))

#Print model
print m

#Print B matrix
print np.round(m.kern.parts[0].B,2)
"""
#Plot W matrix
pb.figure()
W = m.kern.parts[0].W
pb.plot(W[0,:],W[1,:],'kx')
for wi_0, wi_1, name_i in zip(W[0,:],W[1,:],d_names):
    pb.text(x = wi_0, y = wi_1, s = name_i)
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
