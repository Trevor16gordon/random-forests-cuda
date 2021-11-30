import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split



df =pd.read_csv('data/IRIS.csv')
X_all = df.iloc[:, 0:4]
y_all = df.iloc[:, 4:]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)



X_train=X_train.to_numpy() 
y_train =y_train.to_numpy()


labels =np.zeros(y_train.shape)
labels_pred =np.zeros(y_train.shape)
classes = ['Iris-setosa','Iris-virginica','Iris-versicolor']

    

X_n=len(X_train)
weights =(1/X_n)*np.ones(X_n)

choices = np.random.choice(range(X_n), p=weights, size=X_n)
X_train_b = X_train[choices]
y_train_b = y_train[choices]
n= 123
d=3
gina_list=[]



for j in range(len(classes)):
    for i in range(len(y_train)):
        if (y_train[i]== classes[j]):
            labels[i]=1
        else:
            labels[i]=0
    split_value = X_train_b[n,d]

    #check greater than
    for i in range(len(y_train_b)):
        if (X_train_b[i,d]>=split_value):
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



    




    
    

        


