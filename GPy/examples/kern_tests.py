import numpy as np
"""
Sparse Gaussian Processes regression with an RBF kernel
"""
import pylab as pb
import numpy as np
import GPy
np.random.seed(2)
pb.ion()
pb.close('all')
N = 7
D = 2
R = 3
M = 5


_X = np.arange(1,N*D +1).reshape(N,D)
I_ = np.array([0,0,1,1,1,2,2])[:,None]
X = np.hstack([I_,_X])

base =  GPy.kern.rbf(2)
#base = GPy.kern.white(2)
#base = GPy.kern.linear(2)

print "B matrix"
print kernel.parts[0].B
kernel = GPy.kern.icm(base,3,0)

#print kernel.K(X) - base.K(_X)
base.K(_X)
print kernel.K(X)
