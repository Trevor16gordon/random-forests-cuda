import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from random import randrange
from sklearn.metrics import confusion_matrix
import pdb

def checkTerminalCase(labels):
    return len(np.unique(labels)) == 1


class RandomForestFromScratch():

    def __init__(self, n_estimators=1, max_depth=3):
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def fit(self, X_train, y_train):
        self.X_n = len(X_train)
        self.d = len(X_train[0])
        self.weights = (1/self.X_n)*np.ones(self.X_n)

        self.dec_trees = []

        for _ in range(self.n_estimators):

            choices = np.random.choice(range(self.X_n), p=self.weights, size=self.X_n)
            X_train_b = X_train[choices]
            y_train_b = y_train[choices]


            dt = DecisionTree(max_depth=self.max_depth)
            dt.fit(X_train_b, y_train_b)
            self.dec_trees.append(dt.root)

    def predict(self, X):

        all_y_pred = []

        for tr in self.dec_trees:
            all_y_pred.append(tr.predict(X))

        all_res = np.array(all_y_pred)
        most_popular = mode(all_res, axis=0).mode
        return most_popular

class DecisionTree():

    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X_train, y_train):
        self.X_n = len(X_train)
        self.d = len(X_train[0])
        self.weights = (1/self.X_n)*np.ones(self.X_n)


        choices = np.random.choice(range(self.X_n), p=self.weights, size=self.X_n)
        X_train_b = X_train[choices]
        y_train_b = y_train[choices]

        item_stack_fifo = []

        item_stack_fifo.append([X_train_b, y_train_b, 0, "start", None])

        max_iters = 100
        tot_count = 0

        root = None

        while item_stack_fifo and (tot_count < max_iters):

            tot_count += 1
            [X_train_b, y_train_b, depth, node_dir,
                parent] = item_stack_fifo.pop(0)
            n, _ = X_train_b.shape

            is_terminal = checkTerminalCase(y_train_b) or ((depth + 1) >= self.max_depth)
            if not is_terminal:

                if n == 0:
                    # print("Not adding a node here")
                    continue

                all_gina_scores = np.zeros((n, self.d))
                # print(f"ewdbwiuebfuwbef n={n} and d ={self.d}")
                for dim in range(self.d):
                    for row in range(n):
                        all_gina_scores[row, dim] = self._calculate_score(X_train_b, y_train_b, dim, row)

                max_index = self._choose_best_score(all_gina_scores)
                decision_bound = X_train_b[max_index[0], max_index[1]]
                X_train_left_b = X_train_b[X_train_b[:,max_index[1]] <= decision_bound, :]
                X_train_right_b = X_train_b[X_train_b[:,max_index[1]] > decision_bound, :]
                y_train_left_b = y_train_b[X_train_b[:,max_index[1]] <= decision_bound, :]
                y_train_right_b = y_train_b[X_train_b[:,max_index[1]] > decision_bound, :]

                if (len(X_train_left_b) == 0) or (len(X_train_right_b) == 0):
                    print("Data didn't split into two. Making terminal here")
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

    def _calculate_score(self, X_train_b, y_train_b, dim, row):

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

    def _choose_best_score(self, all_gina_scores):
        max_list=[]
        max_list=np.flatnonzero(all_gina_scores == np.max(all_gina_scores))
        max_list=unravel_index(max_list, all_gina_scores.shape)
        max_index_id=randrange(len(max_list[0]))    
        max_index=[max_list[0][max_index_id], max_list[1][max_index_id]]
        return (max_index)


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


if __name__ == "__main__":
    df =pd.read_csv('data/IRIS.csv')
    X_all = df.iloc[:, 0:4]
    y_all = df.iloc[:, 4:]
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)
    rf = RandomForestFromScratch(max_depth=3)
    rf.fit(X_train.to_numpy(), y_train.to_numpy())
    print("fit completed")
    rf.root.depth_first_print()
    predicts=rf.root.predict(X_test.to_numpy())
    confusion=confusion_matrix(predicts,y_test.to_numpy(),labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
    print(confusion)
