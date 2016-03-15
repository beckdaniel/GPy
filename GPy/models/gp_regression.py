# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPRegression(GP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    :param Norm normalizer: [False]
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Gaussian(variance=noise_var)

        super(GPRegression, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

    def predict_reciprocal(self, X, deg_gauss_hermite=20, gh=False):
        """
        Predict the reciprocal of the response variable according
        to a squared error loss. This is *not* equivalent to the
        mean because the reciprocal is not a linear transformation.
        """
        mu, var = GP._raw_predict(self, X)
        # now push through likelihood
        mean, var = self.likelihood.predictive_values(mu, var)

        if not gh:
            return (1 / mean) + (var / (mean ** 3))

        std = np.sqrt(var)
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:, None]
        gh_weights = gh_weights[None, :]
        arg1 = gh_samples.dot(std.T) * np.sqrt(2)
        arg2 = np.ones(shape=gh_samples.shape).dot(mean.T)
        # this is where we change (term / sqrt(pi) to its reciprocal
        rec_mean = gh_weights.dot((1 / (arg1 + arg2)) / np.sqrt(np.pi))

        return rec_mean
