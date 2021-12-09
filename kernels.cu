"""
__global__ calculate_gina_scores(float* impurity_scores,float* X_train,float* y_train,const int  unique_classes_len,const int l,const int w){
    int Dim = threadIdx.x+blockIdx.x*blockDim.x;
    int Row = threadIdx.y+blockIdx.y*blockDim.y;
    if(Dim < w && Row < l){
        float split_value =X_train[Row * w+ Dim];

        //Getting the groups
        int length1=0;
        int length2=0;
        int *group1,*group2;
        group1=(int*)malloc(l*sizeof(int));
        group2=(int*)malloc(l*sizeof(int));

        for(int i =0;i<l;i++){
            if(X_train[i * w+Dim] >= split_value){
                group1+length1 = y_train[i]; 
                length1++
            }
            else{
                group2+length2 = y_train[i]; 
                length2++
            }
        }

        int p1 = length1/(length1+length2);
        int p2 = length2/(length1+length2);

        int group1_counts[20] = {0};//Max of 20 dimensions which can be increased
        group2_counts =group1_counts;
        int sum1=0;
        int sum2=0;

        if(length1 > 0){
            for(int i=0;i<length1;i++){
                group1_counts[*(group1+i)]++;
            }
            for(int i=0;i<unique_classes_len;i++){
                sum1+=group1_counts[i]*group1_counts[i];
            }
            sum1=sum1/(length1*length1);
        }
        if(length2 > 0){
            for(int i=0;i<length2;i++){
                group2_counts[*(group2+i)]++;
            }
            for(int i=0;i<unique_classes_len;i++){
                sum2+=group2_counts[i]*group2_counts[i];
            }
            sum2=sum2/(length2*length2);
        }
        impurity = p1*sum1+p2*sum2;
        // Write our new pixel value out
        impurity_scores[Row * w + Dim] = (impurity);

    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ find_best_gina_score(float* max_value,float* index,float* all_gina_scores, const int row,const int dim){


}
"""