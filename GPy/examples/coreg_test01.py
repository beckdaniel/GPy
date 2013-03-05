import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
import sys

#My functions
sys.path.append('../../../playground/malaria')
import useful

route = '../../../playground/malaria/drafts/draft01_20130225_uai/'

pb.ion()
pb.close('all')

X = np.linspace(0,3,10)[:,None]
Xlist = [np.hstack([np.repeat(i,10)[:,None],X]) for i in range(3)]
Ylist = [np.sin(X)+np.random.rand(10)[:,None]]
Ylist.append(np.sin(2*X)+np.random.rand(10)[:,None])
Ylist.append(np.cos(X)+np.random.rand(10)[:,None])
likelihoods = [GPy.likelihoods.Gaussian(y,normalize=False) for y in Ylist]

pb.plot(X,Ylist[0],'b-')
pb.plot(X,Ylist[1],'r-')
pb.plot(X,Ylist[2],'g-')

base_rbf = GPy.kern.rbf(1)
base_white = GPy.kern.white(1)
R = len(Ylist)
Dw = 2
icm_rbf = GPy.kern.icm(base_rbf,R,index=0,Dw=Dw)
icm_white = GPy.kern.cor_white(base_white,R,index=0,Dw=Dw)
kernel = icm_rbf + icm_white

#Inducing inputs
Z = np.linspace(0,1,5)[:,None]

m = GPy.models.mGP(Xlist, likelihoods, kernel, normalize_X=True,normalize_Y=True)
m.ensure_default_constraints()
m.constrain_positive('kappa')
m.unconstrain('icm*.*var')
m.constrain_fixed('icm*.*var',1)
m.scale_factor=100

if hasattr(m,'Z'):
    m.constrain_fixed('iip',m.Z[:m._M,1].flatten())
m.set('len',1)
m.set('W',.01*np.random.rand(R*Dw))

print m.checkgrad(verbose=1)
m.optimize()
print m


#Plots
Xnew = np.linspace(0,3,100)[:,None]
for i in range(3):
    subs = 310 + i
    pb.subplot(subs)
    Xtest = np.hstack([np.repeat(i,100)[:,None],Xnew])
    mean,var,lower,upper = m.predict(Xtest)
    GPy.util.plot.gpplot(Xtest[:,1],mean,lower,upper)
    pb.plot(X,Ylist[i],'kx',mew = 1.5)

#B matrix
print m.kern.parts[0].B


