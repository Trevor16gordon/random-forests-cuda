
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
    else{
        scan_array[t]=0;
        ii_array[t]=123456789;
    }


    if(start+blockDim.x+t <len){
        scan_array[blockDim.x+t]=input[start+blockDim.x+t];
        ii_array[blockDim.x+t]=index[start+blockDim.x+t];
    }
    else{
        scan_array[blockDim.x+t]=0;
        ii_array[blockDim.x+t]=123456789;
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
