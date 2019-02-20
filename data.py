from common_imports import *
import files
from shutil import move
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

def move_to_folder(srcls, des):
    [move(s, des) for s in srcls]

def scan(folder, format=None, save_path=None, relative=False):
    """
    for simplicity, assume top levels are all folders, bottom levels have less than len//2 files
    :param folder:
    :param format:
    :param save_path:
    :param relative: if True, folder is relative
    :return:
    """
    level = 0 # how many intermediate folders
    if relative:
        parent = files.abs_path(folder)
        logging.warning(f'check if the absolute path of the folder to scan is {parent}')
    else : parent = folder
    all_files = list()
    level_folders = list()
    dfs_path = list()
    while True: # traverse
        if len(level_folders)==level: # meaning children haven't been listed
            children = files.list_dir(parent)

            # check level
            is_dir = [files.is_dir(x) for x in children]
            n_folders = np.sum(is_dir)
            if n_folders <= len(children)//2: # assume it is bottom level
                is_file = negate_bool_list(is_dir)
                if format is not None:
                    is_file = [x.endswith(format) for x in children]
                f = np.asarray(children)[is_file]
                if len(f)>0: all_files.append(f)
                # try to ascend
                parent = dfs_path.pop()
                level = level-1
                continue

            else:
                # attempting to descend
                level_folders.append(children)
        if len(level_folders[level])>0: # there is something left to search, descend
            dfs_path.append(parent)  # record only parents
            parent = level_folders[level].pop() # take the last folder
            level = level + 1
        else:
            level_folders.pop() # pop an empty level
            # try to ascend
            if len(dfs_path)>0:
                parent = dfs_path.pop()
                level = level - 1
            else: break
    if save_path is None: logging.info(all_files)
    else:
        assert save_path.endswith('csv'),'save_path must be csv file'
        paths = np.concatenate(all_files).reshape((-1,1))
        pd.DataFrame(paths, columns=['path']).to_csv(save_path, index=False)



# manipulate data
def negate_bool_array(arr):
    return np.logical_not(arr)

def negate_bool_list(ls):
    return [not x for x in ls]

def ceil_by_factor(a, factor):
    # quo = a // factor
    rem = a % factor
    return a+(factor-rem)

def array_map(arr, fn):
    return np.array(list(map(fn,arr)))

if __name__ == '__main__':
    scan('/usr/whz/bmp_train_data',None,'/usr/whz/EDVSRGAN_root/data_list.csv')
