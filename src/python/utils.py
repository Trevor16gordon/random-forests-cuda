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
    std = np.random.uniform(1, 10, size=(num_classes,num_dimensions))
    
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



class TimingObject():
    """Class to ensure consistent timing information across different tests
    """

    def __init__(self, time, mem_transfer_included, gpu_or_naive, sub_function, num_rows, num_cols):
        """Constructer

        Args:
            time (float): Time in seconds
            mem_transfer_included (bool): Whether the timing includes memory transfer
            gpu_or_naive (str): Generally whether this timing contain
            sub_function (str): choose_best_score or calculate_split_scores or split_data etc
            num_rows (int): Data size
            num_cols (int): Data size
        """
        
        self.store = {
            "time": time,
            "mem_transfer_included": mem_transfer_included,
            "gpu_or_naive": gpu_or_naive,
            "sub_function": sub_function,
            "num_rows": num_rows,
            "num_cols": num_cols
        }

    def __iter__(self):
        return iter(self.store)
    
    def __len__(self):
        return len(self.store)

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, val):
        self.store[key] = val
