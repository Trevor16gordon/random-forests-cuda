"""
__global__ calculate_gina_scores(float* impurity_sscores,float* X_train,float* y_train,float* unique_classes,const int row,const int dim){
    Row=threadIdx.x+
    int split_value = X_train[row_x*dim+dim_x];
    extern __shared__ int *G;
    for(int i =0;i<row;i++){
        if(X_train[i*dim+dim_x]>=split_value){

        }
        else{

        }
    }
}
"""