# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


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
N =400
M =5


######################################
## 1 dimensional example

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(N,1))
Y = np.sin(X)+np.random.randn(N,1)*0.05
X = np.arange(0,N*5,5)[:,None] #remove
Y = np.sin(X/400.) + np.random.randn(N,1)*0.05

# construct kernel
rbf =  GPy.kern.rbf(1)
noise = GPy.kern.white(1)
kernel = rbf + noise

# create simple GP model
m = GPy.models.sparse_GP_regression(X, Y, kernel, M=M,normalize_Y=True)

m.constrain_positive('(variance|lengthscale|precision)')
m.constrain_fixed('iip',m.Z.flatten())
m.checkgrad(verbose=1)
m.optimize('tnc', messages = 1)
pb.subplot(211)
m.plot_f()
pb.subplot(212)
m.plot()
