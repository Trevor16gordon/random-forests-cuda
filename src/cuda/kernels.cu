"""
__global__ calculate_gina_scores(float* impurity_sscores,float* X_train,float* y_train,const int unique_classes,const int row,const int dim){
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
#define BLOCKSIZE 1024
//send an array of indices called index(start from 0 to l-1) which is the 2nd argument in here from python itself.
__global__ find_best_gina_score(float* index,float* all_gina_scores, const int l,const int w){
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
           if(partialSum[t] < partialSum[t+stride]){
               partialSum[t]=partialSum[t+stride];
               index[t]=index[t+stride];
           }
        }            
    }
    //This returns max value and index at the 1st index i.e. 0 in all_gina_scores and index matrices respectively
}
"""