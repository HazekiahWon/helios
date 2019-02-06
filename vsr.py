import exp
from common_imports import *

def offline_performance(a_b_pairs, ds_names, sp_border):
    """

    :param a_b_pairs: list of tuple, each tuple: pattern string for dataset_a's ground truth, one for prediction
    :param ds_names: list of dataset names
    :return: list of list, each list corresponds to psnrs of one dataset
    >>> a_fmt = r'/usr/whz/EDVSRGAN_root/VSR-DUF-master/inputs/G/{}/*.png'
    >>> b_fmt = r'/usr/whz/EDVSRGAN_root/VSR-DUF-master/results/52L/G/{}/*.png'
    >>> datasets = ['calendar','foliage','city','walk']
    >>> offline_performance([(a_fmt.format(x),b_fmt.format(x)) for x in datasets], datasets, sp_border=8)
    """
    rets = [exp.parallel_psnr_compare_tf(*pair, 'png', batch=None, sp_border=sp_border) for pair in a_b_pairs]  # list of dict
    all_psnrs = list()
    psnres = list()
    with tf.Session() as sess:
        for ret, name in zip(rets, ds_names):
            print(name)
            it = ret['it']
            [sess.run(i.initializer) for i in it]
            psnrs = list()
            while True:
                try:
                    psnrs.append(sess.run(ret['op']))
                except tf.errors.OutOfRangeError:
                    break
            m = np.mean(psnrs)
            psnres.append(psnrs)
            print(f'{m}->{psnrs}')
            all_psnrs.append(m)
    print(np.mean(all_psnrs))
    return psnres

def detailize_data_file(file_path, save_path, level=2): # specific for edvsr
    df = pd.read_csv(file_path).sort_values('path')
    for i in range(level):
        df[f'level_{i}'] = df['path'].map(lambda x: x.split(r'/')[-3+i])
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    detailize_data_file('/usr/whz/EDVSRGAN_root/data_list.csv','/usr/whz/EDVSRGAN_root/dtable.csv')