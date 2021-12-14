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
        return
    
    def get_source_module(self):
        # kernel code wrapper
        kernelwrapper = """
        // Helps in the calculation of the gina scores
        __global__ void calculate_gina_scores(float* impurity_scores,float* X_train,int* y_train,const int unique_classes,const int l,const int w){
            int Dim = threadIdx.y+blockIdx.y*blockDim.y;
            int Row = threadIdx.x+blockIdx.x*blockDim.x;

            if(Dim < w && Row < l){
                float split_value =X_train[Row * w+ Dim];

                float group1_counts[20] = {0};//Max of 20 dimensions which can be increased
                float group2_counts[20] = {0};
                float length1=0;
                float length2=0;
                float sum1=0;
                float sum2=0;

                for(int i=0;i<l;i++){
                    if(X_train[i* w+ Dim]>=split_value){
                        //Belongs to group 1
                        group1_counts[y_train[i]]++;
                        length1++;
                    }
                    else{
                        //Belongs to group 2
                        group2_counts[y_train[i]]++;
                        length2++;
                    }
                }
                float p1 = length1/(length1+length2);
                float p2 = length2/(length1+length2);

                if(length1 > 0){
                    for(int i=0;i<unique_classes;i++){
                        sum1+=(group1_counts[i]*group1_counts[i])/(length1*length1);
                    }
                }
                if(length2 > 0){
                    for(int i=0;i<unique_classes;i++){
                        sum2+=(group2_counts[i]*group2_counts[i])/(length2*length2);
                    }
                }

                float impurity = p1*sum1+p2*sum2;
                // Write our new pixel value out
                impurity_scores[Row * w + Dim] =impurity;

            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Finds the max value of all the gina impurity scores that we have calculated
        #define BLOCKSIZE 1024
        __global__ void find_best_gina_score(int* index, float* all_gina_scores, const int len){
            //loading segment of data in local memory
            __shared__ float scan_array[2*BLOCKSIZE];
            __shared__ float ii_array[2*BLOCKSIZE];
            unsigned int t =threadIdx.x;
            unsigned int start=2*blockIdx.x*blockDim.x;

            if(start+t <len){
                scan_array[t]=all_gina_scores[start+t];
                ii_array[t]=index[start+t];
            }
            if(start+blockDim.x+t <len){
                scan_array[blockDim.x+t]=all_gina_scores[start+blockDim.x+t];
                ii_array[blockDim.x+t]=index[start+blockDim.x+t];
            }

            for (unsigned int stride = blockDim.x;stride > 0; stride /= 2){
                __syncthreads();
                if (t < stride){
                  
                  if(scan_array[t] < scan_array[t+stride]){
                      scan_array[t]=scan_array[t+stride];
                      ii_array[t]= ii_array[t+stride];
                  }
                }            
            }
            __syncthreads();
            if(threadIdx.x==0){
                index[blockIdx.x] = ii_array[threadIdx.x];
                all_gina_scores[blockIdx.x] =scan_array[threadIdx.x];
            }
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
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        #define BLOCKSIZE 1024
        __global__ void reduction_scan(float* input,int* index,float* auxSum,int* auxIndex,int len){
            //loading segment of data in local memory
            __shared__ float scan_array[2*BLOCKSIZE];
            __shared__ float ii_array[2*BLOCKSIZE];
            unsigned int t =threadIdx.x;
            unsigned int start=2*blockIdx.x*blockDim.x;

            if(start+t <len){
                scan_array[t]=input[start+t];
                ii_array[t]=index[start+t];
            }
            if(start+blockDim.x+t <len){
                scan_array[blockDim.x+t]=input[start+blockDim.x+t];
                ii_array[blockDim.x+t]=index[start+blockDim.x+t];
            }

            for (unsigned int stride = blockDim.x;stride > 0; stride /= 2){
                __syncthreads();
                if (t < stride){
                  
                  if(scan_array[t] < scan_array[t+stride]){
                      scan_array[t]=scan_array[t+stride];
                      ii_array[t]= ii_array[t+stride];
                  }
                }            
            }
            __syncthreads();
            if(threadIdx.x==0){
                auxIndex[blockIdx.x] = ii_array[threadIdx.x];
                auxSum[blockIdx.x] =scan_array[threadIdx.x];
            }
        }
        """
        return SourceModule(kernelwrapper)
         
    def calculate_score(self, X_train_b, y_train_b):
        
        
        if not isinstance(X_train_b, np.ndarray):
            raise Exception("X_train_b needs to be np.array")
        if not isinstance(y_train_b, np.ndarray):
            raise Exception("y_train_b needs to be np.array")
            
        y_train_b = y_train_b.astype(np.float32)
        X_train_b = X_train_b.astype(np.float32)
            
        # Implement CUDA function
        

        unique_classes = np.unique(y_train_b)
        #Making categorical data into integers
        for j,label in enumerate(unique_classes):
            y_train_b[y_train_b == label]=j
        
        #Fetch the kernel
        start =cuda.Event()
        end = cuda.Event()

        calculate_scores=self.mod.get_function("calculate_gina_scores")

        #Converting to 32 bit
        X_train_b = X_train_b.astype(np.float32)
        y_train_b =y_train_b.astype(np.int32)

        unique_classes=np.array(len(unique_classes)).astype(np.int32)
        row = np.array(X_train_b.shape[0]).astype(np.int32)
        dim = np.array(X_train_b.shape[1]).astype(np.int32)

        #Grid and block dimensions
        blocksize=32
#         blockDim=(blocksize,X_train_b.shape[1],1)
        blockDim=(blocksize, blocksize, 1)
        gridDim =(X_train_b.shape[0]//blocksize+1, X_train_b.shape[1]//blocksize+1, 1)

        #Memory allocation
        X_train_b_gpu = gpuarray.to_gpu(X_train_b)
        y_train_b_gpu = gpuarray.to_gpu(y_train_b)  
        impurity_scores_gpu = gpuarray.zeros_like(X_train_b_gpu)
        
        #run and time the kernel
        start.record()
        print(blockDim)
        print(gridDim)
        calculate_scores(impurity_scores_gpu,
                         X_train_b_gpu,
                         y_train_b_gpu,
                         unique_classes,
                         row,
                         dim,
                         block=blockDim,
                         grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        impurity_scores = impurity_scores_gpu.get()
        return impurity_scores

    # choose_best_score(self, all_gina_scores: np.array):
    #     pycuda.gpuarray.max(
    
    # def choose_best_score(self, all_gina_scores: np.array):
        
    #     if not isinstance(all_gina_scores, np.ndarray):
    #         raise Exception("all_gina_scores needs to be np.array")
    #     all_gina_scores = all_gina_scores.astype(np.float32)
        
    #     all_gina_scores_flatten = all_gina_scores.flatten()
        
    #     current_scores = all_gina_scores_flatten
    #     current_indexes = np.array([i for i in range(all_gina_scores_flatten.shape[0])])
    #     current_indexes = current_indexes.astype(np.int32)
        
    #     t_total = 0
    #     count = 0
    #     block_size = 1024
    #     while True:
            
    #         new_current_indexes, new_current_scores, t = self.choose_best_score_recurs(current_scores, current_indexes)
    #         t_total += t
    #         count += 1
            
            
    #         if len(current_indexes) <= block_size:
    #             print(f"breaking becuase len(current_indexes) is {len(current_indexes)}")
    #             break
                
    #         current_scores, current_indexes = new_current_scores, new_current_indexes
    #         print(f"index Pass- {count} and current index-{len(current_indexes)}")
            
            
    #     max_index=unravel_index(int(current_indexes[0]), all_gina_scores.shape)
    #     return max_index
    
    

    # def choose_best_score_recurs(self, all_gina_scores_flatten, current_indexes):
    #     #Unravel the matrix
        
    #     self.mod = self.get_source_module()
        

    #     #Fetch the kernel 
    #     # start =cuda.Event()
    #     # end = cuda.Event()

    #     find_best_gina_score=self.mod.get_function("find_best_gina_score")
    #     #setting up the eindex matrix
    #     index = current_indexes


    #     #Grid and block dimensions
    #     block_size = 1024
    #     blockDim=(block_size,1,1)
    #     num_blocks = all_gina_scores_flatten.shape[0]//block_size+1
    #     gridDim =(num_blocks,1,1)
    #     print(f"num_blocks {num_blocks}")

    #     #Converting to 32 bit
    #     row =np.float32(all_gina_scores_flatten.shape[0])
    #     all_gina_scores_flatten=all_gina_scores_flatten.astype(np.float32)

    #     #memory allocation
    #     all_gina_scores_gpu=gpuarray.to_gpu(all_gina_scores_flatten)
    #     index=index.astype(np.int32)
    #     index_gpu=gpuarray.to_gpu(index)

    #     #run and time the kernel
    #     # start.record()
    #     find_best_gina_score(index_gpu,all_gina_scores_gpu,row,block=blockDim,grid=gridDim)

    #     # Wait for the event to complete
    #     # end.record()
    #     # end.synchronize()
    #     # time = start.time_till(end)

    #     #Fetch the impurity scores
    #     index=index_gpu.get()
    #     gina_scores=all_gina_scores_gpu.get()
        
        
        
    #     #max_index=unravel_index()

    #     return index[:num_blocks], gina_scores[:num_blocks], 0

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float):
        
        if not isinstance(X, np.ndarray):
            raise Exception("X needs to be np.array")
        if not isinstance(y, np.ndarray):
            raise Exception("y needs to be np.array")

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
        X_32 = X.astype(np.float32)
        row = np.array([X.shape[0]], dtype=np.int32)
        col = np.array([X.shape[1]], dtype=np.int32)
        bound = np.array([bound], dtype=np.int32)
        dim =np.array([dim], dtype=np.int32)
        #Memory allocation
        X_gpu = gpuarray.to_gpu(X_32) 
        labels_gpu = gpuarray.to_gpu(labels)

        #run and time the kernel
        start.record()
        split_data(labels_gpu,X_gpu,bound,dim,row,col,block=blockDim,grid=gridDim)

        # Wait for the event to complete
        end.record()
        end.synchronize()
        time = start.time_till(end)

        #Fetch the impurity scores
        labels = labels_gpu.get().reshape(-1,)

        #code for splitting the child
#         print(labels==0)
        y_l = y[labels==1]
        y_r = y[labels==0]
#         print(y_l,y_r)
#         print(X[0][:])
#         print(f"X.shape is {X.shape}")
#         print(f"labels.shape is {labels.shape}")
        X_l = X[labels==1,:]
        X_r = X[labels==0,:]
        return (X_l, y_l, X_r, y_r)

    def choose_best_score(self,all_gina_scores: np.array):
        a_cud = gpuarray.to_gpu(all_gina_scores)
        max_val_gpu = pycuda.gpuarray.max(a_cud)
        max_val = max_val_gpu.get()
        return np.where(all_gina_scores == max_val)

    def choose_best_score2(self,all_gina_scores: np.array):
      if not isinstance(all_gina_scores, np.ndarray):
            raise Exception("all_gina_scores needs to be np.array")
      #intialize cuda events
      start =cuda.Event()
      end = cuda.Event()
      #fetching the kernel
      scan = self.mod.get_function("reduction_scan")

      #converting the 2d array into 1d for reduction
      all_gina_scores = all_gina_scores.astype(np.float32)
      all_gina_scores_flatten = all_gina_scores.flatten()

      current_indexes = np.array([i for i in range(all_gina_scores_flatten.shape[0])])
      current_indexes = current_indexes.astype(np.int32)

      # Get grid and block dim
      BLOCKSIZE=1024
      blockDim  = (BLOCKSIZE, 1, 1)
      gridDim   = (len(current_indexes)// BLOCKSIZE+1, 1, 1)

      #Making necessary sizes of all the auxillary blocks and sums
      aux_length=len(current_indexes)//BLOCKSIZE+1
      auxSum=np.zeros(aux_length)
      auxSum=np.float32(auxSum)
      auxSum2=np.zeros(32)
      auxSum2=np.float32(auxSum2)
      auxSum3=np.zeros(1)
      auxSum3=np.float32(auxSum3)
      

      auxIndex=np.zeros(aux_length)
      auxIndex=np.int32(auxIndex)
      auxIndex2=np.zeros(32)
      auxIndex2=np.int32(auxIndex2)
      auxIndex3=np.zeros(1)
      auxIndex3=np.int32(auxIndex3)
      aux_length=np.float32(aux_length)
      
      #Sending the info from h2d
      input_gpu = gpuarray.to_gpu(all_gina_scores_flatten)
      index_gpu = gpuarray.to_gpu(current_indexes)
      auxSum_gpu = gpuarray.to_gpu(auxSum)
      auxSum3_gpu = gpuarray.to_gpu(auxSum3)
      auxSum2_gpu = gpuarray.to_gpu(auxSum2)

      auxIndex_gpu = gpuarray.to_gpu(auxIndex)
      auxIndex3_gpu = gpuarray.to_gpu(auxIndex3)
      auxIndex2_gpu = gpuarray.to_gpu(auxIndex2)
      #kernel call
      start.record()
      scan(input_gpu,index_gpu,auxSum_gpu,auxIndex_gpu,np.int32(len(current_indexes)),block=blockDim,grid=gridDim)
      scan(auxSum_gpu,auxIndex_gpu,auxSum2_gpu,auxIndex2_gpu,aux_length,block=blockDim,grid=(32,1,1))
      scan(auxSum2_gpu,auxIndex2_gpu,auxSum3_gpu,auxIndex3_gpu,np.int32(aux_length//BLOCKSIZE+1),block=blockDim,grid=(1,1,1))

      # Wait for the event to complete
      end.record()
      end.synchronize()
      time = start.time_till(end)

      # Fetch result from device to host
      auxIndex3=auxIndex3_gpu.get()
      print(auxIndex3)
      max_index=unravel_index(int(auxIndex3), all_gina_scores.shape)
      return max_index