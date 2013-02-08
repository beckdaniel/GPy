# Test: all districst

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime

pb.ion()
pb.close('all')

R = 3

# Malaria data
def string2date(string):
    year = int(string[0:2])
    month = int(string[3:5])
    day = int(string[6:8])
    return 2000+year, month, day

filename='../../../playground/malaria/allMalariaData'
df=shelve.open(filename)

districts = df.keys()
districts = districts[:R]

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
likelihoods = []


#distribution = GPy.likelihoods.likelihood_functions.Poisson()
for y in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(y))
    #likelihoods.append(GPy.likelihoods.EP(y,distribution))

rbf = GPy.kern.rbf(1)
noise = GPy.kern.white(1)
base = rbf + noise
kernel = GPy.kern.icm(base,R,index=0)

m = GPy.models.multioutput_GP(X_list=X_list,likelihood_list=likelihoods,kernel=kernel,M_i = 10)
m.ensure_default_constraints()
m.unconstrain('rbf_var')
m.constrain_fixed('rbf_var',1.)
m.constrain_positive('kappa')
m.constrain_positive('W')
m.constrain_fixed('iip',m.Z[:,m.input_cols].flatten())
m.set('len',100)
m.set('W',9)
print m
print m.checkgrad(verbose=True)
#m.update_likelihood_approximation()
#m.optimize()
#m.plot()
print m
