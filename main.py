import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.python.cuda_utils import DecisionTreeCudaUtils
from src.python.random_forest import DecisionTreeCudaBaise
from src.python.utils import generate_random_data


# Time test

res = []

for total_data in reversed([10, 100, 1000]):#, 10000]):

  # Setup models
  dtu = DecisionTreeCudaUtils()
  dt_cuda = DecisionTreeCudaBaise(max_depth=4)
  dt_cuda.calculate_split_scores = dtu.calculate_score
  dt_cuda.choose_best_score = dtu.choose_best_score
  dt_cuda.split_data = dtu.split_data

  print("Creating the training data")
  X_train, y_train = generate_random_data(num_dimensions=10, num_samples=total_data, num_classes=3, random_state=13)

  print("Fitting Reference")
  
  cl = RandomForestClassifier(n_estimators=1, max_depth=4)
  # cl = tree.DecisionTreeClassifier(max_depth=3)
  t0 = time.time()
  cl.fit(X_train, y_train)
  t1 = time.time()
  res.append({
      "name": "reference",
      "time": t1 - t0,
      "total_data": total_data*10
  })

  print("Fitting RF Basic")

  t0 = time.time()
  dt_cuda.fit(X_train, y_train)
  t1 = time.time()
  res.append({
      "name": "basic",
      "time": t1 - t0,
      "total_data": total_data*10
  })

print(pd.DataFrame(res))