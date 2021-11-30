import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split



class RandomForestFromScratch():

    def __init__(self, n_estimators=1, max_depth=3):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        pass

    def fit(self, X_train, y_train):
        self.X_n = len(X_train)
        self.d = len(X_train[0])
        self.weights = (1/self.X_n)*np.ones(self.X_n)

        for iter in range(self.n_estimators):

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

                if n == 0:
                    print("Not adding a node here")
                    continue

                all_gina_scores = np.zeros((n, self.d))
                print(f"ewdbwiuebfuwbef n={n} and d ={self.d}")
                for dim in range(self.d):
                    for row in range(n):
                        print(f"X_train_b={X_train_b} and its shape is {X_train_b.shape} and dim is {dim} and row is {row} and y_train_b {y_train_b}")
                        print(f"row{row} dim{dim}")
                        all_gina_scores[row, dim] = self._calculate_score(X_train_b, y_train_b, dim, row)
                        print(f"row{row} dim{dim}")
                        print("hello2")


                best_dim, best_split_val, best_split_val_index = self._choose_best_score(
                    all_gina_scores)

                print(f"best_dim {best_dim}, best_split_val {best_split_val}, best_split_val_index {best_split_val_index} ")

                print(f"X_train_b.shape {X_train_b.shape} best_split_val_index {best_split_val_index}")
                decision_boundary = X_train_b[best_split_val_index, best_dim]

                print(f"node_dir {node_dir} current_depth: {depth}, best_dim {best_dim} best_split_val {best_split_val} decision_boundary {decision_boundary}")

                node = TreeNode(
                    dim=best_dim, val=decision_boundary, depth=depth)

                if parent is not None:
                    if node_dir == "left":
                        parent.left = node
                    else:
                        parent.right = node
                else:
                    root = node

                X_train_left_b = X_train_b[X_train_b[:,best_dim] <= decision_boundary, :]
                X_train_right_b = X_train_b[X_train_b[:,best_dim] > decision_boundary, :]
                y_train_left_b = y_train_b[X_train_b[:,best_dim] <= decision_boundary, :]
                y_train_right_b = y_train_b[X_train_b[:,best_dim] > decision_boundary, :]

                if (depth + 1) < self.max_depth:
                    item_stack_fifo.append(
                        [X_train_left_b, y_train_left_b, depth+1, "left", node])
                    item_stack_fifo.append(
                        [X_train_right_b, y_train_right_b, depth+1, "right", node])

            root.depth_first_print()

    def _calculate_score(self, X_train_b, y_train_b, dim, row):

        gina_list=[]
        classes = ['Iris-setosa','Iris-virginica','Iris-versicolor']
        labels =np.zeros(y_train_b.shape)
        labels_pred =np.zeros(y_train_b.shape)
        for j in range(len(classes)):
            for i in range(len(y_train)):
                if (y_train_b[i]== classes[j]):
                    labels[i]=1
                else:
                    labels[i]=0
            split_value = X_train_b[dim,row]

            #check greater than
            for i in range(len(y_train_b)):
                if (X_train_b[i,row]>=split_value):
                    labels_pred[i]=1
                else:
                    labels_pred[i]=0
            count=0
            for i in range(len(y_train_b)):
                if(labels[i]==labels_pred[i]):
                    count+=1
            print(count)

            #checking the condition
            if(count>=(len(y_train_b)-count)):
                gina_list.append([1,count,classes[j]])
            else:
                gina_list.append([0,(len(y_train_b)-count),classes[j]])
        print(gina_list); 
        print([gina_list[i][1] for i in range(len(classes))])     
        max=np.argmax([gina_list[i][1] for i in range(len(classes))])
        print(max)
        print(gina_list[:][max]) 
        return gina_list[:][max]

    def _choose_best_score(self, all_gina_scores):
        best_split_val_index, best_dim = unravel_index(
            all_gina_scores.argmax(), all_gina_scores.shape)

        best_split_val = all_gina_scores[best_split_val_index, best_dim]
        return (best_dim, best_split_val, best_split_val_index)
        pass


class TreeNode:

    def __init__(self, dim=0, val=0, left=None, right=None, depth=None, is_terminal=False, terminal_val=None):
        self.dim = dim
        self.val = val
        self.left = left
        self.right = right
        self.depth = depth
        self.is_terminal = is_terminal
        self.terminal_val = terminal_val

    def depth_first_print(self):
        print(
            f"Node depth {self.depth} self.dim {self.dim} self.val {self.val}")
        if self.left:
            self.left.depth_first_print()

        if self.right:
            self.right.depth_first_print()

    def predict(self, X):

        test_val = X[:, self.dim]
        if test_val >= self.val:
            return self.right(X)
        else:
            return self.left(X)



df =pd.read_csv('data/IRIS.csv')
X_all = df.iloc[:, 0:4]
y_all = df.iloc[:, 4:]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)
rf = RandomForestFromScratch(max_depth=3)
rf.fit(X_train.to_numpy(), y_train.to_numpy())