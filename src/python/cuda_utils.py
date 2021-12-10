import pandas as pd
import numpy as np
import time
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
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
        kernelwrapper = """
        // Helps in the calculation of the gina scores
        __global__ calculate_gina_scores(float* impurity_scores,float* X_train,float* y_train,const int unique_classes,const int l,const int w){
            int Dim = threadIdx.x+blockIdx.x*blockDim.x;
            int Row = threadIdx.y+blockIdx.y*blockDim.y;
            if(Dim < w && Row < l){
                float split_value =X_train[Row * w+ Dim];

                int group1_counts[20] = {0};//Max of 20 dimensions which can be increased
                group2_counts =group1_counts;
                int length1=0;
                int length2=0;
                int sum1=0;
                int sum2=0;

                for(int i=0;i<l;i++){
                    if(X_train[i* w+ Dim]>=split_value){
                        //Belongs to group 1
                        group1_counts[y[i]]++;
                        length1++;
                    }
                    else{
                        //Belongs to group 2
                        group2_counts[y[i]]++;
                        length2++;
                    }
                }
                int p1 = length1/(length1+length2);
                int p2 = length2/(length1+length2);

                if(length1 > 0){
                    for(int i=0;i<unique_classes;i++){
                        sum1+=(group1_counts*group1_counts)/(length1*length1);
                    }
                }
                if(length2 > 0){
                    for(int i=0;i<unique_classes;i++){
                        sum2+=(group2_counts*group2_counts)/(length2*length2);
                    }
                }

                impurity = p1*sum1+p2*sum2;
                // Write our new pixel value out
                impurity_scores[Row * w + Dim] = (impurity);

            }
        }



        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Finds the max value of all the gina impurity scores that we have calculated
        #define BLOCKSIZE 1024
        __global__ void find_best_gina_score(float* index,float* all_gina_scores, const int len){
            //loading segment of data in local memory
            __shared__ float scan_array[2*BLOCKSIZE];
            unsigned int t =threadIdx.x;
            unsigned int start=2*blockIdx.x*blockDim.x;

            if(start+t <len){
                scan_array[t]=all_gina_scores[start+t];
            }
            else{
                scan_array[t]=0;
            }

            if(start+blockDim.x+t <len){
                scan_array[blockDim.x+t]=all_gina_scores[start+blockDim.x+t];
            }
            else{
                scan_array[blockDim.x+t]=0;
            }

            for (unsigned int stride = blockDim.x;stride > 0; stride /= 2){
                __syncthreads();
                if (t < stride){
                  if(scan_array[t] < scan_array[t+stride]){
                      scan_array[t]=scan_array[t+stride];
                      index[t]=index[t+stride];
                  }
                }            
            }
            //This returns max value and index at the 1st index i.e. 0 in all_gina_scores and index matrices respectively
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //takes the input matrix X and returns the labels for l or r
        __global__ void split_data(float* label,float* X, const int bound,const int dim, const int l,const int w){
            int Row = threadIdx.x+blockIdx.x*blockDim.x;
            if(Row < l){
                if(X[Row*w+dim] <= bound){
                    label[Row]=1;
                }
                else{
                    label[Row]=0;
                }
            }           
        }
        """
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

        find_best_gina_score=self.mod.get_function("find_best_gina_score")
        #setting up the eindex matrix
        index=[i for i in range(all_gina_scores_flatten.shape[0])]
        index=np.array(index)


        #Grid and block dimensions
        blockDim=(1024,1,1)
        gridDim =(all_gina_scores_flatten.shape[0]//1024+1,1,1)

        #Converting to 32 bit
        row =np.float32(all_gina_scores_flatten.shape[0])
        all_gina_scores_flatten=all_gina_scores_flatten.astype(np.float32)

        #memory allocation
        all_gina_scores_gpu=gpuArray.to_gpu(all_gina_scores_flatten)
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
        max_index=unravel_index(int(index[0]),all_gina_scores.shape)

        return(max_index,time)

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float):

        # Implement CUDA function
        #Fetch the kernel
        start =cuda.Event()
        end = cuda.Event()

        split_data=self.mod.get_function("split_data")
        #Grid and block dimensions
        blockDim=(1024,1,1)
        gridDim =(X.shape[0]//1024+1,1,1)
        #Converting to 32 bit
        labels = np.zeros(y.shape).astype(np.float32)
        X = X.astype(np.float32)
        row = np.array([X.shape[0]], dtype=np.int32)
        col = np.array([X.shape[1]], dtype=np.int32)
        bound = np.array([bound], dtype=np.int32)
        dim =np.array([dim], dtype=np.int32)
        #Memory allocation
        X_gpu = gpuArray.to_gpu(X) 
        labels_gpu = gpuArray.to_gpu(labels)

        #run and time the kernel
        print("starting the kernel")
        start.record()
        split_data(labels_gpu,X_gpu,bound,dim,row,col,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        print("ending the kernel")
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        labels = labels_gpu.get()

        #code for splitting the child
        X_l = X[labels==0, :]
        X_r = X[labels==1, :]
        y_l = y[labels==0]
        y_r = y[labels==1]
        return (X_l, y_l, X_r, y_r)