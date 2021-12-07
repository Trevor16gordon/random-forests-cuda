import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split


def checkTerminalCase(lst):
    return len(set(lst)) == 1



df =pd.read_csv('data/IRIS.csv')
X_all = df.iloc[:, 0:4]
y_all = df.iloc[:, 4:]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)



X_train=X_train.to_numpy() 
y_train =y_train.to_numpy()


labels =np.zeros(y_train.shape)
labels_pred =np.zeros(y_train.shape)
classes = ['Iris-setosa','Iris-virginica','Iris-versicolor']

    

X_n=15
weights =(1/X_n)*np.ones(X_n)

choices = np.random.choice(range(X_n), p=weights, size=X_n)
X_train_b = X_train[choices]
y_train_b = y_train[choices]
n= 1
d=2
gina_list=[]

for j in range(len(classes)):

    #create the labels
    labels = y_train_b==classes[j]
    labels =labels[:,0].astype('uint8')
    print(np.sum(labels))

    print(f"Labels are {labels}")

    #get the split value
    split_value = X_train_b[n,d]
    print(f"the split value is {split_value}")

    #check greater than
    labels_pred=(X_train_b[:,d]>=split_value)
    labels_pred=labels_pred.astype('uint8')
    print(f'the predicted labels are {labels_pred}')

    #get the count
    count=np.sum(np.array([(labels[i]==labels_pred[i]) & (labels[i]==1) for i in range(len(y_train_b))]).astype('uint8'))
    print(count)

    #check less than
    labels_pred=(X_train_b[:,d]<split_value)
    labels_pred=labels_pred.astype('uint8')
    print(f'the 2nd  predicted labels are {labels_pred}')

    #get the count
    count1=np.sum(np.array([(labels[i]==labels_pred[i]) & (labels[i]==1) for i in range(len(y_train_b))]).astype('uint8'))
    print(count1)

    if count>count1:
        gina_list.append([1,count,classes[j]])
    else:
        gina_list.append([0,count1,classes[j]])

print(gina_list); 
print([gina_list[i][1] for i in range(len(classes))])     
max=np.argmax([gina_list[i][1] for i in range(len(classes))])
print(max)
print(gina_list[:][max]) 



    




    
    

        


