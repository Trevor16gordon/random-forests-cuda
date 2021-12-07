import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split
from random import randrange
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

                is_terminal = checkTerminalCase(y_train_b) or ((depth + 1) >= self.max_depth)
                if not is_terminal:

                    if n == 0:
                        # print("Not adding a node here")
                        continue

                    all_gina_scores = np.zeros((n, self.d))
                    all_gina_info = np.zeros((n, self.d), dtype="object")
                    # print(f"ewdbwiuebfuwbef n={n} and d ={self.d}")
                    for dim in range(self.d):
                        for row in range(n):
                            # print(f"X_train_b={X_train_b} and its shape is {X_train_b.shape} and dim is {dim} and row is {row} and y_train_b {y_train_b}")
                            # print(f"row{row} dim{dim}")
                            all_gina_scores[row, dim],all_gina_info[row,dim]= self._calculate_score(X_train_b, y_train_b, dim, row)
                    #print(all_gina_scores)


                    best_split_val,max_score_info,max_index = self._choose_best_score(all_gina_scores,all_gina_info)
                    # print(f"best_split_value {best_split_val}, max_score_info {max_score_info}, max_index {max_index}")

                    # print(f"X_train_b.shape {X_train_b.shape}")
                    decision_boundary = X_train_b[max_index[0], max_index[1]]

                    # print(f"node_dir {node_dir} current_depth: {depth}, best_dim {max_index[1]} best_split_val {best_split_val} decision_boundary {decision_boundary}")

                    

                    X_train_left_b = X_train_b[X_train_b[:,max_index[1]] <= decision_boundary, :]
                    X_train_right_b = X_train_b[X_train_b[:,max_index[1]] > decision_boundary, :]
                    y_train_left_b = y_train_b[X_train_b[:,max_index[1]] <= decision_boundary, :]
                    y_train_right_b = y_train_b[X_train_b[:,max_index[1]] > decision_boundary, :]

                    if (len(X_train_left_b) == 0) or (len(X_train_right_b) == 0):
                        print("Data didn't split into two. Making terminal here")
                        is_terminal = True

                if not is_terminal:
                    node = TreeNode(dim=max_index[1], val=decision_boundary, depth=depth)

                else: # Terminal Case
                    (unique, counts) = np.unique(y_train_b, return_counts=True)
                    sorted_count_values = [x for _, x in sorted(zip(counts, unique), reverse=True)]
                    sorted_counts = [_ for _, x in sorted(zip(counts, unique), reverse=True)]
                    highest_count_label = sorted_count_values[0]
                    # print(f"highest_count_label is {highest_count_label}")
                    # print(sorted_count_values)
                    # print(sorted_counts)
                    # pdb.set_trace()
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
                    # print(f"Terminating {node_dir} node")

            root.depth_first_print()
            self.root = root

    def _calculate_score(self, X_train_b, y_train_b, dim, row):

        gina_list=[]
        classes = ['Iris-setosa','Iris-virginica','Iris-versicolor']
        labels =np.zeros(y_train_b.shape)
        labels_pred =np.zeros(y_train_b.shape)
        for j in range(len(classes)):

            #create the labels
            labels = y_train_b==classes[j]
            labels =labels[:,0].astype('uint8')
            # print(np.sum(labels))

            #print(f"Labels are {labels}")

            #get the split value
            split_value = X_train_b[row,dim]
            #print(f"the split value is {split_value}")

            #check greater than
            labels_pred=(X_train_b[:,dim]>=split_value)
            labels_pred=labels_pred.astype('uint8')
            #print(f'the predicted labels are {labels_pred}')

            #get the count
            count=np.sum(np.array([(labels[i]==labels_pred[i]) & (labels[i]==1) for i in range(len(y_train_b))]).astype('uint8'))
            #print(count)

            #check less than
            labels_pred=(X_train_b[:,dim]<split_value)
            labels_pred=labels_pred.astype('uint8')
            #print(f'the 2nd  predicted labels are {labels_pred}')

            #get the count
            count1=np.sum(np.array([(labels[i]==labels_pred[i]) & (labels[i]==1) for i in range(len(y_train_b))]).astype('uint8'))
            #print(count1)

            if count>count1:
                gina_list.append( {"sign": 1, "num_correct": count, "class": classes[j]})
            else:
                gina_list.append( {"sign": 0, "num_correct": count1, "class": classes[j]})

        # print(gina_list); 
        # print([gina_list[i]["num_correct"] for i in range(len(gina_list))])     
        max=np.argmax([gina_list[i]["num_correct"] for i in range(len(gina_list))])
        # print(max)
        # print(gina_list[:][max]) 
        return gina_list[:][max]["num_correct"],gina_list[:][max]

    def _choose_best_score(self, all_gina_scores, all_gina_score_info):
        # print("Checking the best score")
        max_list=[]
        max_list=np.flatnonzero(all_gina_scores == np.max(all_gina_scores))
        max_list=unravel_index(max_list,all_gina_scores.shape)
        max_index=randrange(len(max_list[0]))
        max_index=[max_list[0][max_index],max_list[1][max_index]]
        
        # print(max_list)
        # print(max_index)
        
        best_split_val = all_gina_scores[max_index[0],max_index[1]]
        max_score_info = all_gina_score_info[max_index[0],max_index[1]]
        return (best_split_val, max_score_info, max_index)


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
        # print(
        #     f"Node depth {self.depth} self.dim {self.dim} self.val {self.val}")

        if self.is_terminal:
            # print(f"Terminal class is {self.terminal_val}")
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



# df =pd.read_csv('data/IRIS.csv')
# X_all = df.iloc[:, 0:4]
# y_all = df.iloc[:, 4:]
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)
# rf = RandomForestFromScratch(max_depth=3)
# rf.fit(X_train.to_numpy(), y_train.to_numpy())
# print("fit completed")
# rf.root.depth_first_print()
