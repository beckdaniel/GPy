"""
Multioutput GP for malaria counts
dataset: ../../../playground/malaria/malaria_data20130213.dat
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

#Define districts to analize
d_names = ['Masindi','Mpigi','Wakiso','Kampala']#,'Mukono','Luwero','Tororo']
d_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == d_i] for d_i in d_names])

#Define input
X_names = ['district','date']
X_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == n_i] for n_i in X_names])
X_list = [malaria_data['data'][d_i][:,X_numbers] for d_i in d_numbers]
for X_i,num_i in zip(X_list,range(len(d_names))):
    X_i[:,0] = np.repeat(num_i,X_i.shape[0]) #Change district number according to data analyzed

#Define output
Y_names = ['incidences']
Y_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == Y_i] for Y_i in Y_names])
Y_list = [malaria_data['data'][d_i][:,Y_numbers] for d_i in d_numbers]
if len(Y_names) > 1:
    for num_i in range(len(d_numbers)):
        Y_list[num_i] = np.vstack(Y_list[num_i])

#Close data file
malaria_data.close()

#Create likelihood
likelihoods = []
for Y_i in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(Y_i))

# Define coreg_kern and base kernels
R = len(d_names)
D = len(X_names) - 1 #-1 if district is in X_names
rbf = GPy.kern.rbf(D)
noise = GPy.kern.white(D)
#bias = GPy.kern.bias(3)
base = rbf + rbf.copy() + noise #+ bias
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

# Define the model
m = GPy.models.mGP(X_list, likelihoods, kernel, normalize_X=True) #NOTE: better to normalize X and Y

# Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_1_var')
m.constrain_fixed('rbf_1_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.unconstrain('rbf_2_var')
m.constrain_fixed('rbf_2_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
m.set('_1_len',.10)
m.set('_2_len',10)
m.set('W',np.random.rand(R*2))

# Optimize
print m.checkgrad(verbose=True)
m.optimize()

# Plots
m.plot()
#for r in range(R):
    #m.plot_HD(input_col=1,output_num=r)
print m
print np.round(m.kern.parts[0].B,2)
