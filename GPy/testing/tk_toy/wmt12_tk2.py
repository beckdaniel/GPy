
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
SUB_TRAIN = 100
SUB_TEST = 10

def filter_root(tree_array):
    for i,tree in enumerate(tree_array):
        tree_array[i][0] = tree[0][6:-2]
    return tree_array


np.set_printoptions(suppress=True)
with open(TREES_TRAIN) as f:
    X_trees_train = np.array([[line] for line in f.readlines()], dtype=object)[:SUB_TRAIN]
    X_trees_train = filter_root(X_trees_train)
with open(TREES_TEST) as f:
    X_trees_test = np.array([[line] for line in f.readlines()], dtype=object)[:SUB_TEST]
    X_trees_test = filter_root(X_trees_test)

Y_train = np.loadtxt(LABELS_TRAIN, ndmin=2)[:SUB_TRAIN]
Y_test = np.loadtxt(LABELS_TEST, ndmin=2)[:SUB_TEST]


tk = GPy.kern.SubsetTreeKernel(_lambda=1, _sigma=1)
m = GPy.models.GPRegression(X_trees_train, Y_train, kernel=tk)

# I suspect that lambda X marginal likelihood is an asymptote so let's fix it for now
m['sstk_lambda'] = 0.1
m.constrain_fixed('sstk_lambda')

# Our focus now is optimize sigma. Minimum depends on noise but it is usually between 0.1 and 0.2.

print "\nOur focus now is optimize sigma. Optimum depends on noise but it is usually between 0.1 and 0.2.\n"

# SETTING 1 - sigma=1
#
# This does not optimize at all.
print "SETTING 1 - This does not optimize at all.\n"
print m
m.optimize(messages=True, max_iters=100)
print m

# SETTING 2 - sigma=0.3
#
# We initialized sigma a bit nearer to the minimum (check LL) but it still does not optimize.

print "SETTING 2 - We initialized sigma a bit nearer to the optimum (check LL) but it still does not optimize.\n"
m['noise_variance'] = 1
m['sstk_sigma'] = 0.3
print m
m.optimize(messages=True, max_iters=100)
print m

# SETTING 3 - sigma=0
#
# Initializing as 0 seems to work though...

print "SETTING 3 - Initializing as 0 seems to work though..."
m['noise_variance'] = 1
m['sstk_sigma'] = 0
print m
m.optimize(messages=True, max_iters=100)
print m

# SETTING 4 - sigma=0 + positive constraint
#
# Putting a positive constraint changes stuff: sigma stays at 0 but noise changes.

print "SETTING 4 - Putting a positive constraint changes stuff: sigma stays at 0 but noise changes."
m['noise_variance'] = 1
m['sstk_sigma'] = 0
m.constrain_positive('sstk_sigma')
print m
m.optimize(messages=True, max_iters=100)
print m


# IPDB, I usually use this to play with the models.
#import ipdb
#ipdb.set_trace()
 
