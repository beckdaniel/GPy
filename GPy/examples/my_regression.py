
import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')

X1 = np.arange(0,100,5)[:,None]
X2 = np.arange(150,250,5)[:,None]
X3 = np.arange(300,400,5)[:,None]
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

#pb.plot(X1,Y1,'kx')
#pb.plot(X2,Y2,'rx')
#pb.plot(X3,Y3,'bx')

base = GPy.kern.rbf(1)
kernel = GPy.kern.icm(base,3,index=0)

#print "Base"
#print base.K(X_)
#print "Multikern"
#print kernel.K(X)


#pb.figure()
# create simple GP model
m = GPy.models.multioutput_GP(X_list,likelihood_list)
# optimize
m.ensure_default_constraints()
m.unconstrain("rbf_var")
m.constrain_fixed("rbf_var",1.)
m.constrain_positive("kappa")
m.constrain_positive("W")
print "Coregionalization matrix"
print m.checkgrad()
m.optimize()
print m.kern.parts[0].B
#pb.subplot(211)
#m.plot_f()
#pb.subplot(212)
m.plot()
print m
#y0,y1 = pb.ylim()
#x0,x1 = pb.xlim()

"""
GP_regression
q = GPy.models.GP(X_,likelihood_,base)
q.constrain_fixed("rbf_var",1)
q.constrain_fixed("noise",30)
q.constrain_fixed("len",60)
#q.ensure_default_constraints()
#print q.checkgrad()
#q.optimize()
#pb.figure()
#q.plot()
#pb.xlim(x0,x1)
#pb.ylim(y0,y1)
print q


Z = np.random.rand(7)[:,None]
I = np.hstack([np.repeat(0,2),np.repeat(1,3),np.repeat(2,2)])[:,None]
Zplus = np.hstack([I,Z])

print m.kern.K(Zplus) - q.kern.K(Z)
"""
