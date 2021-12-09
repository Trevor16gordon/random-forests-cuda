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
        #Grid and block dimensions
        blockDim=(1024,1024,1)
        gridDim =(self.d//1024+1,self.d//1024+1,1)

        #Converting to 32 bit
        X_train_b = X_train_b.astype(np.float32)
        y_train_b =y_train_b.astype(np.float32)
        unique_classes=unique_classes.astype(np.float32)
        row = X_train_b.shape[0].astype(np.float32)
        dim = X_train_b.shape[1].astype(np.float32)

        #Memory allocation
        X_train_b_gpu = gpuArray.to_gpu(X_train_b)
        y_train_b_gpu = gpuArray.to_gpu(y_train_b)
        unique_classes_gpu=gpuArray.to_gpu(unique_classes)
        impurity_scores_gpu = gpuArray.zeros_like(X_train_b_gpu)



        #run and time the kernel
        start.record()
        calculate_gina_scores(impurity_scores_gpu,X_train_b_gpu,y_train_b_gpu,unique_classes_gpu,row,dim,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        impurity_scores = impurity_scores_gpu.get()

        #################Check with trevor about gina_info requirement
        return impurity_scores,time
    
    def choose_best_score(self, scores: np.array):
        #Fetch the kernel
        start =cuda.Event()
        end = cuda.Event()

        find_best_gina_score=self.mod.get("find_best_gina_score")

        #Grid and block dimensions
        blockDim=(1024,1024,1)
        gridDim =(self.d//1024+1,self.d//1024+1,1)

        #Converting to 32 bit
        row,dim =np.float32(all_gina_scores.shape)
        all_gina_scores=all_gina_scores.astype(np.float32)
        

        #memory allocation
        all_gina_scores_gpu=gpuArray.to_gpu(all_gina_scores)

        max_value =np.zeros(1)
        max_value =max_value.astype(np.float32)
        max_value_gpu=gpuArray.to_gpu(max_value)

        index=np.zeros(2)
        index=index.astype(np.float32)
        index_gpu=gpuArray.to_gpu(index)


        #run and time the kernel
        start.record()
        find_best_gina_score(max_value_gpu,index_gpu,all_gina_scores_gpu,row,dim,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        index=index_gpu.get()

        return(index,time)

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float):
        # Implement CUDA function
        pass
    