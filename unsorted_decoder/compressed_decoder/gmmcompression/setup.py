#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("fkmixture",
                             sources=["fkmixture.pyx","covariance_c.c","component_c.c","mixture_c.c"],
                             libraries=['gsl', 'gslcblas', 'm'],
                             include_dirs=[numpy.get_include()])],
)
