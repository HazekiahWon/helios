import numpy as np
class Metrics:
    idx2name = ['psnr','mse']
    name2idx = {n:idx for idx,n in enumerate(idx2name)}
    def __init__(self, m):
        self.ops = m # optim, psnr, mse,[..]
        self.collections = list()
        self.colname2idx = dict()

    def get_psnr_op(self): return self.ops[0]

    def get_all_ops(self): return self.ops

    def create_collections_for(self, names): # must ensure names in name2idx
        if isinstance(names, str): names = [names]
        for idx,name in enumerate(names):
            self.collections.append(Collections(name, ['def']))
            self.colname2idx[name] = idx

    def create_collections_for_all(self):
        self.create_collections_for(self.idx2name)

    def append_collections_for_all(self, values):
        for n,v in zip(self.idx2name,values):
            self.collections[self.colname2idx[n]].append_to_collection('def',v)

    def get_collections_for_all(self):
        return [self.collections[self.colname2idx[n]].get_collection('def') for n in self.idx2name]



class Collections:
    def __init__(self, name, cols=tuple()):
        self.name = name # optim, psnr, [..]
        self.collection = list()
        self.pointers = list()
        self.indices = dict() # can only use string to reference
        self.free_nodes = list()
        for col in cols:
            self.init_collection(col)

    def collection_name2idx(self, name):
        try:
            idx = self.indices[name]
        except:
            raise Exception(f'collection {name} does not exist')
        return idx



    def init_collection(self, name):
        end = start = len(self.collection)
        if len(self.free_nodes)==0:
            idx = len(self.pointers)
            self.indices[name] = idx
            self.pointers.append([start, end])
        else:
            idx = self.free_nodes.pop()
            self.indices[name] = idx
            self.pointers[idx] = [start, end]

    def append_to_collection(self, names, value):
        """

        :param names:
        :param value:
        :return:
        """

        self.collection.append(value)
        if isinstance(names, str): names = [names]
        for n in names:
            idx = self.collection_name2idx(n)
            self.pointers[idx][1] += 1 # inc end

    def reinit_collection(self, name):
        end = start = len(self.collection)
        self.pointers[self.collection_name2idx(name)] = [start,end]

    def drop_collection(self, name):
        idx = self.indices.pop(name)
        self.free_nodes.append(idx)

    def clear_all(self):
        self.collection.clear()
        for idx in self.indices.values():
            self.pointers[idx] = [0,0]

    def get_collection(self, name):
        idx = self.collection_name2idx(name)
        s,e = self.pointers[idx]
        return self.collection[s:e]

    def unravel_bundles_in_collection(self, name):
        ret_list = self.get_collection(name)
        pl, ll = zip(*ret_list)  # pl: a tuple of array
        pl = np.concatenate(list(pl))
        m = np.sum(pl) / np.sum(ll)
        return m, pl

    def collection_mean(self, name, reinit=True):
        m = np.mean(self.get_collection(name))
        if reinit: self.reinit_collection(name)
        return m
