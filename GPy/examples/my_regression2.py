
import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')

X1 = np.hstack([np.arange(0,100,5)[:,None],np.arange(0,100,5)[:,None]])
X2 = np.hstack([np.arange(150,250,5)[:,None],np.arange(0,100,5)[:,None]])
X3 = np.hstack([np.arange(300,400,5)[:,None],np.arange(0,100,5)[:,None]])
X_list_ = [X1,X2,X3]
X_list = [np.hstack([np.ones((20,1))*i,x]) for i,x in zip(range(3),X_list_)]
X_ = np.vstack(X_list_)
X = np.vstack(X_list)

F1 = np.round(np.sin(10*X1/18.) + .1*X1) + np.arange(5,25)[:,None]
F2 = np.round(5*np.sin(X2/8.) + .5*X2) + np.arange(5,25)[:,None]
F3 = 200 + 2*np.cos(X3/8.) + -.5*X3 + np.arange(5,25)[:,None]

E1 = np.random.randint(-5,5,20)[:,None]
E2 = 4*np.random.randint(-5,5,20)[:,None]
E3 = np.random.randint(-5,5,20)[:,None]

Y1 = F1 + E1
Y2 = F2 + E2
Y3 = F3 + E3
Y_ = np.vstack([Y1,Y2,Y3])
likelihood_ = GPy.likelihoods.Gaussian(Y_)
likelihood1 = GPy.likelihoods.Gaussian(Y1)
likelihood2 = GPy.likelihoods.Gaussian(Y2)
likelihood3 = GPy.likelihoods.Gaussian(Y3)
likelihood_list = [likelihood1,likelihood2,likelihood3]

base = GPy.kern.rbf(2)
kernel = GPy.kern.icm(base,3,index=0)
"""
m = GPy.models.multioutput_GP(X_list,likelihood_list,M_i = 3)
# optimize
m.ensure_default_constraints()
m.unconstrain('rbf_var')
m.constrain_fixed('rbf_var',1.)
m.constrain_positive('kappa')
m.constrain_positive('W')
m.constrain_fixed('iip',m.Z[:,m.input_cols].flatten())
#m.set('W',1)
print m.get('W')
m.set('len',2.)
print m.checkgrad(verbose=1)
m.optimize()
print 'Coregionalization matrix'
print m.kern.parts[0].B
print m

print "mGP"
"""
q = GPy.models.mGP(X_list,likelihood_list)
q.ensure_default_constraints()
q.unconstrain('rbf_var')
q.constrain_fixed('rbf_var',1.)
q.constrain_positive('kappa')
q.constrain_positive('W')
q.set('len',60)
print q.checkgrad(verbose=1)
q.optimize()
print 'Coregionalization matrix'
print q.kern.parts[0].B
print q
q.plot_f()
