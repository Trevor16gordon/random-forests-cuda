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
from numpy import unravel_index

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

        unique_classes = np.unique(y_train_b)
        #Making categorical data into integers
        for j,label in enumerate(unique_classes):
            y_train_b[y_train_b == label]=j
        
        #Fetch the kernel
        start =cuda.Event()
        end = cuda.Event()

        calculate_gina_scores=self.mod.get("calculate_gina_scores")

        #Grid and block dimensions
        blockDim=(1024,X_train_b.shape[1],1)
        gridDim =(X_train_b.shape[0]//1024+1,X_train_b.shape[1]//1024+1,1)

        #Converting to 32 bit
        X_train_b = X_train_b.astype(np.float32)
        y_train_b =y_train_b.astype(np.float32)

        unique_classes=len(unique_classes).astype(np.float32)
        row = X_train_b.shape[0].astype(np.float32)
        dim = X_train_b.shape[1].astype(np.float32)

        #Memory allocation
        X_train_b_gpu = gpuArray.to_gpu(X_train_b)
        y_train_b_gpu = gpuArray.to_gpu(y_train_b)  
        impurity_scores_gpu = gpuArray.zeros_like(X_train_b_gpu)
        
        #run and time the kernel
        start.record()
        calculate_gina_scores(impurity_scores_gpu,X_train_b_gpu,y_train_b_gpu,unique_classes,row,dim,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        impurity_scores = impurity_scores_gpu.get()
        return impurity_scores,time
    
    def choose_best_score(self, all_gina_scores: np.array):
        #Unravel the matrix
        all_gina_scores_flatten =all_gina_scores.flatten()
        #Fetch the kernel 
        start =cuda.Event()
        end = cuda.Event()

        find_best_gina_score=self.mod.get("find_best_gina_score")

        #Grid and block dimensions
        blockDim=(1024,1,1)
        gridDim =(all_gina_scores_flatten.shape[0]//1024+1,1,1)

        #Converting to 32 bit
        row =np.float32(all_gina_scores.shape[0])
        all_gina_scores_flatten=all_gina_scores_flatten.astype(np.float32)

        #memory allocation
        all_gina_scores_gpu=gpuArray.to_gpu(all_gina_scores_flatten)

        index=[i for i in range(row)]
        index=index.astype(np.float32)
        index_gpu=gpuArray.to_gpu(index)

        #run and time the kernel
        start.record()
        find_best_gina_score(index_gpu,all_gina_scores_gpu,row,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        index=index_gpu.get()
        max_index=unravel_index(index,all_gina_scores.shape)

        return(max_index,time)

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float):
        # Implement CUDA function
        #Fetch the kernel
        start =cuda.Event()
        end = cuda.Event()

        split_data=self.mod.get("split_data")

        #Grid and block dimensions
        blockDim=(1024,X.shape[1],1)
        gridDim =(X.shape[0]//1024+1,X.shape[1]//1024+1,1)

        #Converting to 32 bit
        labels = np.zeros(y).astype(np.float32)
        X = X.astype(np.float32)

        row = X.shape[0].astype(np.float32)
        col = X.shape[1].astype(np.float32)
        bound = bound.astype(np.float32)
        dim =dim.astype(np.float32)

        #Memory allocation
        X_gpu = gpuArray.to_gpu(X) 
        labels=np.zeros(y).astype(np.float32)
        labels_gpu = gpuArray.to_gpu(labels)

        #run and time the kernel
        start.record()
        split_data(labels_gpu,X_gpu,bound,dim,row,col,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        labels = labels_gpu.get()

        #code for splitting the child
        X_l = X[labels==0, :]
        X_r = X[labels==1, :]
        y_l = y[labels==0, :]
        y_r = y[labels==1, :]
        return (X_l, y_l, X_r, y_r)












        pass
    