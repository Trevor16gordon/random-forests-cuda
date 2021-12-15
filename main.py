import pycuda
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.python.cuda_utils import DecisionTreeCudaUtils
from src.python.random_forest import DecisionTreeCudaBase, DecisionTreeNativePython
from src.python.utils import generate_random_data


# Time test

res = []
dtu = DecisionTreeCudaUtils()
for num_dimensions in [10, 100, 1000]:
    for num_rows in [10, 100, 1000, 10000]:

        if (num_dimensions == 1000) and (num_rows == 1000):
            continue

        # Setup models
        pycuda.tools.clear_context_caches()
        del dtu
        dtu = DecisionTreeCudaUtils()
        dt_cuda = DecisionTreeCudaBase(max_depth=4)
        dt_python = DecisionTreeNativePython(max_depth=4)


        
        print("Creating the training data")
        X_train, y_train = generate_random_data(num_dimensions=num_dimensions, num_samples=num_rows, num_classes=3, random_state=13)

        print("Fitting Reference")
        
        cl = RandomForestClassifier(n_estimators=1, max_depth=4)
        # cl = tree.DecisionTreeClassifier(max_depth=3)
        t0 = time.time()
        cl.fit(X_train, y_train)
        t1 = time.time()
        res.append({
            "name": "sklearn",
            "time": t1 - t0,
            "total_data": num_rows*num_dimensions,
            "num_rows": num_rows,
            "num_dimensions": num_dimensions,
        })

        print("Fitting GPU cuda")

        t0 = time.time()
        dt_cuda.fit(X_train, y_train)
        t1 = time.time()
        res.append({
            "name": "gpu-basic",
            "time": t1 - t0,
            "total_data": num_rows*num_dimensions,
            "num_dimensions": num_dimensions,
        })


        if (num_dimensions < 1000) and (num_rows < 10000):
            print("Fitting RF python implementation")

            t0 = time.time()
            dt_python.fit(X_train, y_train)
            t1 = time.time()
            res.append({
                "name": "gpu-basic",
                "time": t1 - t0,
                "total_data": num_rows*num_dimensions,
                "num_dimensions": num_dimensions,
            })

print(pd.DataFrame(res))
