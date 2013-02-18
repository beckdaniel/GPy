"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/malaria_data20130213.dat
B matrix controls the relation between districts
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
stations = malaria_data['stations']

#Define districts to analize
d_names = ['Mubende','Masindi','Mbarara','Kampala','Kasese']
#['Mubende','Nakasongola']#,'Kamuli','Kampala','Mukono','Luwero','Tororo']
d_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == d_i] for d_i in d_names])
if len(d_names) > len(d_numbers):
    print 'Warning: some districts were not found in malaria_data'

#Define output
Y_names = ['incidences']
Y_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == Y_i] for Y_i in Y_names])
Y_list = [malaria_data['data'][d_i][:,Y_numbers] for d_i in d_numbers]
if len(Y_names) > 1:
    for num_i in range(len(d_numbers)):
        Y_list[num_i] = np.vstack(Y_list[num_i])

#Define input
X_names = ['district','time']#,'time']#,'rain','ndvi','humidity_06','humidity_12','rain','temperature_min','temperature_max']
X_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == n_i] for n_i in X_names])
X_list = [malaria_data['data'][d_i][:,X_numbers] for d_i in d_numbers]
for X_i,num_i in zip(X_list,range(len(d_names))):
    X_i[:,0] = np.repeat(num_i,X_i.shape[0]) #Change district number according to data analyzed

#Remove last observations of all the districts
p_int = 60
Y_star_list = []
for i in range(len(Y_list)):
    Y_star_list.append(Y_list[i][p_int:])
    Y_list[i] = Y_list[i][:p_int]
#Y_star = Y_list[0][p_int:]
#Y_list[0] = Y_list[0][:p_int]
X_star_list = []
for i in range(len(X_list)):
    X_star_list.append(X_list[i][p_int:,:])
    X_list[i] = X_list[i][:p_int,:]
#X_star = X_list[0][p_int:,:] #for future prediction
#X_list[0] = X_list[0][:p_int,:] #for model fitting

#Close data file
malaria_data.close()

#Create likelihood
likelihoods = []
for Y_i in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(Y_i))

#Define coreg_kern and base kernels
R = len(d_names)
D = len(X_names) - 1 #-1 if district is in X_names
rbf = GPy.kern.rbf(D)
noise = GPy.kern.white(D)
base = rbf + rbf.copy() + noise
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

"""
Regression
"""
q_rbf = GPy.kern.rbf(1)
q_noise = GPy.kern.white(1)
q_base = q_rbf + q_rbf.copy() + q_noise
q = GPy.models.GP(X_list[0][:,1][:,None],likelihoods[0],base,normalize_X=True)
q.ensure_default_constraints()
q.optimize()
q.plot()
m, var, lower, upper = q.predict(X_star_list[0][:,1][:,None])
GPy.util.plot.gpplot(X_star[:,1],m, lower, upper)
pb.plot(X_star[:,1],Y_star,'rx',mew=1.5)
pb.figure()


#Define model
m = GPy.models.mGP(X_list, likelihoods, kernel, normalize_X=True,normalize_Y=True)

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
"""
for os,on in zip(m.Xos,range(len(m.Xos))):
    for vn in range(len(X_names)-1):
        pb.figure()
        m.plot_HD(which_input=vn,which_data=os)
        pb.title('%s: %s' %(d_names[on],X_names[vn+1]))

"""
m.plot(which_data = m.Xos[0])

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
"""
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
index = 0

#Nnew = X_list.shape[0]
#Xnew_list = X_list
#Xnew_list[0] = X_star
#Xnew = np.vstack([Xnew_list])

#Xu = self.X[which_data,:] * self._Xstd + self._Xmean #NOTE self.X are the normalized values now
#Xu, _index = self._index_off(Xu)
#index = int(_index[0,0])
#Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
#I_ = np.repeat(index,resolution or 200)[:,None] #repeat the first index number
#Xnew = self._index_on(Xnew,I_)
m, var, lower, upper = m.predict(X_star)
#pb.figure()
GPy.util.plot.gpplot(X_star_list[0][:,1],m, lower, upper)
#pb.plot(Xu,m.likelihoods[index].data,'kx',mew=1.5)
pb.plot(X_star[:,1],Y_star,'rx',mew=1.5)
#ymin,ymax = min(np.append(m.likelihoods[index].data,lower)), max(np.append(m.likelihoods[index].data,upper))
#ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
#pb.xlim(xmin,xmax)
#pb.ylim(ymin,ymax)


