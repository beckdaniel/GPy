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

base1 =  GPy.kern.rbf(2)
base2 = GPy.kern.white(2)
#base = GPy.kern.linear(2)

print base1.K(_X)
print base2.K(_X)
print (base1+base2).K(_X)
#print base1.add(base2).K(_X) - (base1+base2).K(_X)

kernel1 = GPy.kern.icm(base1,3,0)
kernel2 = GPy.kern.icm(base2,3,0)

print "B matrix"
#print kernel1.parts[0].B

print kernel1.K(X)# - base1.K(_X)

#kernel3 = kernel1.add(kernel2)
print 'kernel 1'
print kernel1.K(X)
print 'kernel 2'
print kernel2.K(X)

kernel3 = kernel1 + kernel2
print 'kernel3'
print kernel3.K(X)

