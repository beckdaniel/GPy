#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


# Version number
version = '0.4.6'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

ext_module = Extension(
    "cy_sst",
    ["GPy/kern/parts/cy_tree.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(name = 'GPy',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "james.hensman@gmail.com",
      description = ("The Gaussian Process Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "http://sheffieldml.github.com/GPy/",
      packages = ['GPy', 'GPy.core', 'GPy.kern', 'GPy.util', 'GPy.models_modules', 'GPy.inference', 'GPy.examples', 'GPy.likelihoods', 'GPy.testing', 'GPy.util.latent_space_visualizations', 'GPy.util.latent_space_visualizations.controllers', 'GPy.likelihoods.noise_models', 'GPy.kern.parts', 'GPy.mappings'],
      package_dir={'GPy': 'GPy'},
      package_data = {'GPy': ['GPy/examples', 'gpy_config.cfg']},
      py_modules = ['GPy.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.6', 'scipy>=0.9','matplotlib>=1.1', 'nose'],
      extras_require = {
        'docs':['Sphinx', 'ipython'],
      },
      classifiers=[
      "License :: OSI Approved :: BSD License"],
      #ext_modules =  [Extension(name = 'GPy.kern.lfmUpsilonf2py',
      #          sources = ['GPy/kern/src/lfmUpsilonf2py.f90'])],
      ext_modules = cythonize("GPy/kern/parts/*.pyx"),
      )
