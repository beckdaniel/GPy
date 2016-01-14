import numpy as np
from .kern import Kern


class StringKernel(Kern):
    """
    String Kernel
    """
    def __init__(self, name='sk'):
        super(StringKernel, self).__init__(1, 1, name)
