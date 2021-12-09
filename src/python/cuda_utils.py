import pandas as pd
import numpy as np
import time
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from scipy.misc import ascent
from scipy import signal
import pdb

cuda.init()
print("Number of CUDA devices available: ", cuda.Device.count())
my_device = cuda.Device(0)
# cc: compute capability
cc = float('%d.%d' % my_device.compute_capability())
print('Device Name: {}'.format(my_device.name()))
print('Compute Capability: {}'.format(cc))
print('Total Device Memory: {} megabytes'.format(my_device.total_memory()//1024**2))


class DecisionTreeCudaUtils():

    def __init__(self):
        """
        Attributes for instance of EncoderDecoder module
        """
        self.mod = self.get_source_module()
        pass
    
    def get_block_and_grid_dim(self, length):
        """Get block and grid dimensions.
        """
        blocksize = 32
        blockDim = (blocksize, 1, 1)
        gridDim = ((length // blocksize) + 1, 1, 1)
        return blockDim, gridDim

    def get_source_module(self):
        # kernel code wrapper
        kernelwrapper = ""
        return SourceModule(kernelwrapper)
    
    def calculate_score(self, X_train_b, y_train_b, dim, row):
        # Implement CUDA function
        pass
    
    def choose_best_score(self, scores: np.array):
        # Implement CUDA function
        pass

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float):
        # Implement CUDA function
        pass
    