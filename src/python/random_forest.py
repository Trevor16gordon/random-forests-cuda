import time
import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from random import randrange
from sklearn.metrics import confusion_matrix
from src.python.utils import TimingObject, generate_random_data
from src.python.cuda_utils import DecisionTreeCudaUtils


class RandomForestFromScratch():

    def __init__(self, n_estimators=1, max_depth=3, use_gpu=False):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.use_gpu = use_gpu

    def fit(self, X_train, y_train):
        start = time.time()
        self.X_n = len(X_train)
        self.d = len(X_train[0])
        self.weights = (1/self.X_n)*np.ones(self.X_n)

        self.dec_trees = []
        self.timing_objs = []

        for _ in range(self.n_estimators):

            choices = np.random.choice(range(self.X_n), p=self.weights, size=self.X_n)
            X_train_b = X_train[choices]
            y_train_b = y_train[choices]

            if self.use_gpu:
                dt = DecisionTreeCudaBase(max_depth=self.max_depth)
            else:
                dt = DecisionTreeNativePython(max_depth=self.max_depth)
            
            timing_objs = dt.fit(X_train_b, y_train_b)
            self.timing_objs.extend(timing_objs)
            self.dec_trees.append(dt.root)

        elapsed = time.time() - start
        time_obj = TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="gpu" if self.use_gpu else "naive",
            sub_function="top_level",
            num_rows=self.X_n,
            num_cols=self.d)
        self.timing_objs.append(time_obj)
        return self.timing_objs

        

    def predict(self, X):

        all_y_pred = []

        for tr in self.dec_trees:
            all_y_pred.append(tr.predict(X))

        all_res = np.array(all_y_pred)
        most_popular = mode(all_res, axis=0).mode
        return most_popular


class DecisionTreeBase():
    """Base decision tree to inherit from

    Returns:
        [type]: [description]
    """

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def check_terminal_case(self, labels, n, d):
        raise NotImplementedError()

    def calculate_split_scores(self, X: np.array, y: np.array) -> np.array:
        raise NotImplementedError()

    def choose_best_score(self, scores: np.array) -> list:
        raise NotImplementedError()

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float) -> tuple:
        raise NotImplementedError()

    def fit(self, X_train, y_train):
        self.X_n = len(X_train)
        self.d = len(X_train[0])
        self.weights = (1/self.X_n)*np.ones(self.X_n)
        item_stack_fifo = []
        item_stack_fifo.append([X_train, y_train, 0, "start", None])
        max_iters = 100
        tot_count = 0
        root = None

        self.timing_objs = []

        while item_stack_fifo and (tot_count < max_iters):

            tot_count += 1
            [X_train_b, y_train_b, depth, node_dir,
                parent] = item_stack_fifo.pop(0)
            n, _ = X_train_b.shape

            all_same_classes, timing_obj = self.check_terminal_case(y_train_b, n, self.d) 
            is_terminal = all_same_classes or ((depth + 1) >= self.max_depth)
            self.timing_objs.append(timing_obj)
            if not is_terminal:

                if n == 0:
                    continue
                
                all_gina_scores, time_objs = self.calculate_split_scores(X_train_b, y_train_b)
                self.timing_objs.extend(time_objs)     
                max_index, time_objs = self.choose_best_score(all_gina_scores)
                self.timing_objs.extend(time_objs)
                decision_bound = X_train_b[max_index[0], max_index[1]]
                
                (X_train_left_b,
                y_train_left_b,
                X_train_right_b,
                y_train_right_b,
                time_objs) = self.split_data(X_train_b, y_train_b, decision_bound, max_index[1])
                self.timing_objs.extend(time_objs)

                if (len(X_train_left_b) == 0) or (len(X_train_right_b) == 0):
                    is_terminal = True

            if not is_terminal:
                node = TreeNode(dim=max_index[1], val=decision_bound, depth=depth)

            # Terminal Case
            else: 
                (unique, counts) = np.unique(y_train_b, return_counts=True)
                sorted_count_values = [x for _, x in sorted(zip(counts, unique), reverse=True)]
                highest_count_label = sorted_count_values[0]
                node = TreeNode(dim=None, val=None, depth=depth, is_terminal=True, terminal_val=highest_count_label)
                

            if parent is not None:
                if node_dir == "left":
                    parent.left = node
                else:
                    parent.right = node
            else:
                root = node

            if not is_terminal:
                
                item_stack_fifo.append([X_train_left_b, y_train_left_b, depth+1, "left", node])
                item_stack_fifo.append([X_train_right_b, y_train_right_b, depth+1, "right", node])
            else:
                pass
        self.root = root
        return self.timing_objs

    
class DecisionTreeNativePython(DecisionTreeBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_terminal_case(self, labels, n, d):
        start = time.time()
        res = len(np.unique(labels)) == 1
        elapsed = time.time() - start
        time_obj = TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="naive",
            sub_function="check_terminal_case",
            num_rows=d, 
            num_cols=n)
        return res, time_obj

    def calculate_split_scores(self, X: np.array, y: np.array) -> np.array:
        start = time.time()
        n, d = X.shape
        all_gina_scores = np.zeros((n, d))
        for dim in range(d):
            for row in range(n):
                all_gina_scores[row, dim] = self._calculate_score(X, y, dim, row)
        elapsed = time.time() - start
        time_obj = TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="naive",
            sub_function="calculate_score",
            num_rows=d, 
            num_cols=n)
        return all_gina_scores, [time_obj]

    def _calculate_score(self, X_train_b, y_train_b, dim, row):
        start = time.time()
        unique_classes = np.unique(y_train_b)

        split_value = X_train_b[row,dim]
        group1 = y_train_b[X_train_b[:, dim]>=split_value]
        group2 = y_train_b[X_train_b[:, dim]<split_value]

        p1 = len(group1) / (len(group1) + len(group2))
        p2 = len(group2) / (len(group1) + len(group2))

        group1_counts = np.zeros(len(unique_classes))
        group2_counts = np.zeros(len(unique_classes))

        if len(group1) > 0:
            for ii, cls_i in enumerate(unique_classes):
                group1_counts[ii] = np.count_nonzero(group1 == cls_i) / len(group1)
        if len(group2) > 0:
            for ii, cls_i in enumerate(unique_classes):
                group2_counts[ii] = np.count_nonzero(group2 == cls_i) / len(group2)

        impurity = p1 * sum([x**2 for x in group1_counts]) + p2 * sum([x**2 for x in group2_counts])
        return impurity

    def choose_best_score(self, scores: np.array) -> list:
        start = time.time()
        n, d = scores.shape
        max_list=[]
        max_list=np.flatnonzero(scores == np.max(scores))
        max_list=unravel_index(max_list, scores.shape)
        max_index_id=randrange(len(max_list[0]))    
        max_index=[max_list[0][max_index_id], max_list[1][max_index_id]]
        elapsed = time.time() - start
        time_obj = TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="naive",
            sub_function="choose_best_score",
            num_rows=d, 
            num_cols=n)
        return max_index, [time_obj]

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float) -> tuple:
        start = time.time()
        n, d = X.shape
        X_l = X[X[:,dim] <= bound, :]
        X_r = X[X[:,dim] > bound, :]
        y_l = y[X[:,dim] <= bound, :]
        y_r = y[X[:,dim] > bound, :]
        elapsed = time.time() - start
        time_obj = TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="naive",
            sub_function="split_data",
            num_rows=d, 
            num_cols=n)
        return X_l, y_l, X_r, y_r, [time_obj]


class DecisionTreeCudaBase(DecisionTreeBase):

    def __init__(self, max_depth):
        super().__init__(max_depth)
        self.cuda_utils = DecisionTreeCudaUtils()

    def check_terminal_case(self, labels, n, d):
        start = time.time()
        res = len(np.unique(labels)) == 1
        elapsed = time.time() - start
        time_obj = TimingObject(
            time=elapsed,
            mem_transfer_included=True, 
            gpu_or_naive="gpu",
            sub_function="check_terminal_case",
            num_rows=d, 
            num_cols=n)
        return res, time_obj

    def calculate_split_scores(self, X: np.array, y: np.array) -> np.array:
        return self.cuda_utils.calculate_score(X, y)

    def choose_best_score(self, scores: np.array) -> list:
        return self.cuda_utils.choose_best_score(scores)

    def split_data(self, X: np.array, y: np.array, bound: float, dim: float) -> tuple:
        return self.cuda_utils.split_data(X, y, bound, dim)


class TreeNode:

    def __init__(self, dim=0, val=0, left=None, right=None, depth=None, is_terminal=False, terminal_val=None):
        self.dim = dim
        self.val = val
        self.left = left
        self.right = right
        self.depth = depth
        self.is_terminal = is_terminal
        self.terminal_val = terminal_val
        self.label_dtype = object

    def depth_first_print(self, output_txt=""):
        if self.is_terminal:
            output_txt += "|   "*self.depth +  "|---" + f" class: {self.terminal_val}\n"
        else:
            output_txt += "|   "*self.depth +  "|---" + f"feature_{self.dim} >={self.val}\n"
        if self.left:
            output_txt = self.left.depth_first_print(output_txt)

        if self.right:
            output_txt = self.right.depth_first_print(output_txt)
        if self.depth == 0:
            print(output_txt)
        return output_txt

    def predict(self, X):

        test_val = X[:, self.dim]

        if not self.is_terminal:
            labels = np.empty((len(test_val)), dtype=self.label_dtype)
            X_r = X[test_val >= self.val, :]
            X_l = X[test_val < self.val, :]
            # print(f"len(X) {len(X)} len(X_r) {len(X_r)} len(X_l) {len(X_l)}")
            labels[test_val >= self.val] = self.right.predict(X_r)
            labels[test_val < self.val] = self.left.predict(X_l)
        else:
            labels = np.full((len(test_val)), self.terminal_val, dtype=self.label_dtype)


        return labels


# if __name__ == "__main__":

X_all, y_all = generate_random_data(num_dimensions=3, num_samples=100, num_classes=3, random_state=13)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)
rf = RandomForestFromScratch(max_depth=3)
rf.fit(X_train, y_train)
print("fit completed")
# rf.root.depth_first_print()
# predicts=rf.predict(X_test.to_numpy())
# confusion=confusion_matrix(predicts,y_test.to_numpy(),labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
# print(confusion)



