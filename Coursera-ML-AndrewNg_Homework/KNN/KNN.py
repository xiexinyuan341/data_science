import numpy as np
import operator

def create_dataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = np.array(['A','A','B','B'])
    return group,labels