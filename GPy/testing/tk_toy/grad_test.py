
import numpy as np
import GPy
import nltk
import sys
import os

TRAIN_DIR = 'train'
TEST_DIR = 'test'
TREES_TRAIN = os.path.join(TRAIN_DIR, 'trees.tsv')
LABELS_TRAIN = os.path.join(TRAIN_DIR, 'labels.tsv')
TREES_TEST = os.path.join(TEST_DIR, 'trees.tsv')
LABELS_TEST = os.path.join(TEST_DIR, 'labels.tsv')

#SUB_TRAIN = int(sys.argv[1])
#SUB_TEST = int(sys.argv[2])
SUB_TRAIN = 10
SUB_TEST = 5

def filter_root(tree_array):
    for i,tree in enumerate(tree_array):
        tree_array[i][0] = tree[0][6:-2]
    return tree_array

def checkgrad(grad, f1, f2, step=1e-4):
    print "ANALYTICAL:"
    print grad
    print "NUMERICAL:"
    print (f1 - f2) / (2 * step)

np.set_printoptions(suppress=True)
with open(TREES_TRAIN) as f:
    X_trees_train = np.array([[line] for line in f.readlines()], dtype=object)[:SUB_TRAIN]
    X_trees_train = filter_root(X_trees_train)
with open(TREES_TEST) as f:
    X_trees_test = np.array([[line] for line in f.readlines()], dtype=object)[:SUB_TEST]
    X_trees_test = filter_root(X_trees_test)

Y_train = np.loadtxt(LABELS_TRAIN, ndmin=2)[:SUB_TRAIN]
Y_test = np.loadtxt(LABELS_TEST, ndmin=2)[:SUB_TEST]


tk = GPy.kern.SubsetTreeKernel(_lambda=0.1, _sigma=1, normalize=False)
m = GPy.models.GPRegression(X_trees_train, Y_train, kernel=tk)

#####################################################
# First, a grad check test on the kernel derivatives.
# We set dL_dK = 1.

step = 1e-4
print m
print "KERNEL GRADS - SUMS"
grads = np.zeros(2)
m.kern.parts[0].dK_dtheta(1, X_trees_train, None, grads)

m['sstk_lambda'] += step
tgt = np.zeros(shape=(SUB_TRAIN, SUB_TRAIN))
m.kern.parts[0].K(X_trees_train, None, tgt)
f1 = np.sum(tgt)
m['sstk_lambda'] -= 2*step
tgt = np.zeros(shape=(SUB_TRAIN, SUB_TRAIN))
m.kern.parts[0].K(X_trees_train, None, tgt)
f2 = np.sum(tgt)
checkgrad(grads[0], f1, f2)

######################################################
# We can also check the matrices instead of just their sum.
print ''
print "KERNEL GRADS - MATRICES"
m['sstk_lambda'] = 0.1
grad = m.kern.parts[0].dlambda
m['sstk_lambda'] += step
f1 = m.kern.parts[0].kernel.K(X_trees_train, None)[0]
m['sstk_lambda'] -= 2*step
f2 = m.kern.parts[0].kernel.K(X_trees_train, None)[0]
checkgrad(grad, f1, f2)

######################################################
# Now multiplying by dL_dK breaks everything
print ''
print 'MULTIPLYING KERNEL GRADS BY dL_dK - SUMS'
m['sstk_lambda'] = 0.1
grad = m.kern.parts[0].dlambda * m.dL_dK
m['sstk_lambda'] += step
f1 = m.kern.parts[0].kernel.K(X_trees_train, None)[0] * m.dL_dK
m['sstk_lambda'] -= 2*step
f2 = m.kern.parts[0].kernel.K(X_trees_train, None)[0] * m.dL_dK
checkgrad(np.sum(grad), np.sum(f1), np.sum(f2))

print ''
print 'MULTIPLYING KERNEL GRADS BY dL_dK - MATRICES'
checkgrad(grad, f1, f2)
