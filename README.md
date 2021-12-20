# random-forests-cuda

This repo contains a random forest implementations from scratch using native python and a cuda GPU implemendation


# Running
- Clone repo
- Change directory to the root
- run ```python main.py```

# Overview of files in the repo
- main.py is an entry point that runs timing tests for fitting
- src/python/random_forest.py
    - class RandomForestFromScratch
        - Interface for running Random Forest models giving the same interface as sklearn Random Forests implementation
        - This class instantiates many different DecisionTreeNativePython or DecisionTreeCudaBase and trains them on random subsets of the training data.
        - When predicting the majority voting class of all the predictor trees are used
    - class DecisionTreeBase
        - Interface for the base Decision tree class giving same interface as sklearn Decision Tree Implemendation
        - Intended to subclass this and overwrite a few specific functions: calculate_split_scores, choose_best_score, split_data
        - This class includes the logic for iteratively training a decision tree by doing a breadth first training on the leaf nodes
        - Training is completed when the maximum depth is reached or leaf nodes only contain a single class
    - class DecisionTreeNativePython
        - calculate_split_scores naive serial implemendation
        - choose_best_score naive serial implemendation
        - split_data naive serial implemendation
    - class DecisionTreeCudaBase
        - calculate_split_scores points to src/utils/cuda_utils.DecisionTreeCudaUtils().split_scores
        - choose_best_score points to src/utils/cuda_utils.DecisionTreeCudaUtils().choose_best_score
        - split_data points to src/utils/cuda_utils.DecisionTreeCudaUtils().split_data  
- src/python/utils/cuda_utils.py
    - class DecisionTreeCudaUtils
        - Intended to be the python interface into cuda functions
        - Written like the structure of our assignments
- src/cuda/kernels.cu
    - Cuda kernel functions written here
- src/python/utils.py
    - Includes generation of variable dimensional gaussian mixture model data for running different timing tests
    - Includes some timing objects to standardize timing across modules

# Results
Results including figures and tabular data are shown in the results folder


