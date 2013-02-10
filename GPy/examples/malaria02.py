"""
Multioutput GP for malaria counts
dataset: ../../../playground/malaria/allMalariaData
The model is applied to  Districts that are geographicaly close to each other
W is a 2 x R matrix
---------------------------------------------
Masindi <-                            Luweero
           Mpigi - Wakiso - Kampala - Mukono
"""

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
pb.ion()
pb.close('all')

def string2date(string):
#Converts a string into a date, this function is needed to format the dataset
    year = int(string[0:2])
    month = int(string[3:5])
    day = int(string[6:8])
    return 2000+year, month, day


# Process Malaria data
filename='../../../playground/malaria/allMalariaData'
df=shelve.open(filename)
#districts = df.keys()
districts = ['Masindi','Mpigi','Wakiso','Kampala','Mukono','Luweero']
X_list = []
Y_list = []
for d,n in zip(districts,range(len(districts))):
    N = len(df[d].keys())
    X = []
    Y = []
    new_date = []
    date = df[d].keys()
    date.sort()
    for i in date:
        correct_value = True
        if df[d][i] is not None:
            year,month,day=string2date(i)
            new_date.append(datetime.date(year,month,day))
            try:
                Y.append(float(df[d][i]))
            except ValueError:
                print "Unexpected value: %s - %s - %s" %(d,i,df[d][i])
                correct_value = False
            except TypeError:
                print "Unexpected value: %s - %s - %s" %(d,i,df[d][i])
                correct_value = False
            if correct_value:
                #newX = (new_date[-1] - new_date[0]).days
                X.append((new_date[-1] - new_date[0]).days)
    assert len(X) == len(Y)
    X = np.array(X)[:,None]
    Y = np.array(Y)[:,None]
    I = np.repeat(n,X.shape[0])[:,None]
    X_list.append(np.hstack([I,X]))
    Y_list.append(Y)

#Set number of districts to work with (i.e. number of outputs)
R = len(districts)

# Define Gaussian likelihood
likelihoods = []
for y in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(y))

# Define the inducing inputs
M = R #NOTE: the model won't work properly if M is different from R
Zindex = [np.repeat(i,M)[:,None] for i in range(len(X_list))]
_Z = [np.linspace(0,1300,M)[:,None] for a in range(M)]
Z_list = [np.hstack([_i,_Z]) for _i,_Z in zip(Zindex,_Z)]

# Define coreg_kern and base kernels
rbf = GPy.kern.rbf(1)
bias = GPy.kern.bias(1)
noise = GPy.kern.white(1)
base = rbf + noise #+ bias
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

# Define the model
m = GPy.models.multioutput_GP(X_list=X_list,likelihood_list=likelihoods,kernel=kernel,Z_list=Z_list,normalize_X=True) #NOTE: better to normalize X and Y

# Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_var')
m.constrain_fixed('rbf_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
m.constrain_positive('W')
m.constrain_fixed('iip',m.Z[:,m.input_cols].flatten()) #No need to optimize this
m.set('len',.1) #NOTE the model works better initializing lengthscale as .1

# Optimize
print m.checkgrad(verbose=True)
m.optimize()

# Plots
m.plot()
print m
print np.round(m.kern.parts[0].B,2)
