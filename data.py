import pickle
import numpy as np
def save_pkl(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    # print(f'save data in {fname}.pkl')

def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)

def load_npy(fname):
    return np.load(fname)

def save_npy(fname,data):
    np.save(fname, data)

def check_dist(a,b,minimum=0.01):
    return np.sum(np.abs(a-b)>minimum)