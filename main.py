import pycuda
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.python.cuda_utils import DecisionTreeCudaUtils
from src.python.random_forest import DecisionTreeCudaBase, DecisionTreeNativePython, RandomForestFromScratch
from src.python.utils import generate_random_data
from src.python.utils import TimingObject

# Time test

all_timing_objs = []

res = []
n_estimators = 1
max_tree_depth = 4
for num_dimensions in [10, 100, 1000]:
    for num_rows in [10, 100, 1000, 10000]:

        if (num_dimensions == 1000) and (num_rows == 1000):
            continue

        # Setup models
        # pycuda.tools.clear_context_caches()
        # del dtu
        # dtu = DecisionTreeCudaUtils()
        dt_cuda = RandomForestFromScratch(n_estimators=n_estimators, use_gpu=True, max_depth=max_tree_depth)
        dt_python = RandomForestFromScratch(n_estimators=n_estimators, use_gpu=False, max_depth=max_tree_depth)

        print("Creating the training data")
        X_train, y_train = generate_random_data(num_dimensions=num_dimensions, num_samples=num_rows, num_classes=3, random_state=13)

        print("Fitting Reference")
        
        cl = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_tree_depth)
        start = time.time()
        cl.fit(X_train, y_train)
        elapsed = time.time() - start
        all_timing_objs.append(TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="sklearn",
            sub_function="top_level",
            num_rows=num_dimensions, 
            num_cols=num_rows))

        print("Fitting GPU cuda")

        timeing_objs = dt_cuda.fit(X_train, y_train)
        all_timing_objs.extend(timeing_objs)
        
        if (num_dimensions < 1000) and (num_rows < 10000):
            print("Fitting RF python implementation")
            timeing_objs = dt_python.fit(X_train, y_train)
            all_timing_objs.extend(timeing_objs)

df = pd.DataFrame([x.store for x in all_timing_objs])

print(df)
df.to_csv("random_forests_gpu_2.csv")
