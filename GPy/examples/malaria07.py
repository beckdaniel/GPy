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

#Define districts to analize
d_names = ['Mubende','Nakasongola']#,'Kamuli']#,'Kampala','Mukono','Luwero','Tororo']
d_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == d_i] for d_i in d_names])

#Define output
Y_names = ['incidences','rain']
Y_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == Y_i] for Y_i in Y_names])
Y_list = []
for output_i in Y_numbers:
    Y_list.append([malaria_data['data'][d_i][:,output_i] for d_i in d_numbers])
    Y_list[-1] = np.hstack(Y_list[-1])[:,None]

#Define input
X_names = ['district','time','ndvi']
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
    likelihoods.append(GPy.likelihoods.Gaussian(Y_i))

#Define coreg_kern and base kernels
R = len(Y_names)
D = len(X_names) - 1 #-1 if district is in X_names
rbf = GPy.kern.rbf(D)
noise = GPy.kern.white(D)
base = rbf + rbf.copy() + noise
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

#Define model
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
m.set('_2_len',.1)
m.set('W',np.random.rand(R*2))

#Optimize
print m.checkgrad(verbose=True)
m.optimize()

#Plots
#m.plot_f()
#m.plot()
for r in range(R):
    for i in range(len(X_names)-1):
        pb.figure()
        m.plot_HD(input_col=i,output_num=r)
        pb.title('%s vs %s' %(Y_names[r],X_names[i+1]))

#Print model
print m

#Print B matrix
print np.round(m.kern.parts[0].B,2)
"""
# Plot W matrix
pb.figure()
W = m.kern.parts[0].W
pb.plot(W[0,:],W[1,:],'kx')
for wi_0, wi_1, name_i in zip(W[0,:],W[1,:],Y_names):
    pb.text(x = wi_0, y = wi_1, s = name_i)
"""
