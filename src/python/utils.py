import numpy as np
from sklearn.utils import shuffle

def generate_random_data(num_dimensions=3, num_samples=100, num_classes=3, random_state=None):
    # Random weighted amounts of each class
    random_states = list(range(random_state, random_state+3+num_classes*num_dimensions))
    np.random.seed(random_states.pop(0))
    cls_probs = np.array([0] + sorted(np.random.uniform(size=(num_classes - 1))) + [1])
    cls_probs = cls_probs[1:] - cls_probs[:-1]
    cls_num_samples = [int(num_samples*p) for p in cls_probs]
    cls_num_samples[0] += num_samples - sum(cls_num_samples)
    np.random.seed(random_states.pop(0))
    means = np.random.normal(1, 10, size=(num_classes,num_dimensions))
    np.random.seed(random_states.pop(0))
    std = np.random.uniform(1, 10, size=(num_classes,num_dimensions))\
    
    X = np.zeros((num_samples, num_dimensions))
    y = np.zeros((num_samples, 1))
    offset = 0
    for class_i in range(num_classes):
        y[offset:offset+cls_num_samples[class_i]] = class_i
        for dim in range(num_dimensions):
            np.random.seed(random_states.pop(0))
            X[offset:offset+cls_num_samples[class_i], dim] = np.random.normal(means[class_i,dim],
                                                                          std[class_i,dim],
                                                                          size=cls_num_samples[class_i])
        offset += cls_num_samples[class_i]
        
    X, y = shuffle(X, y)
    return X, y