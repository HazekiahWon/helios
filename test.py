from common_imports import *
# import sys
print(sys.path)
import data,vsr
def filter_ds(paths, seq_len, amp=4.):
    read_path,val_path,train_path = paths
    df = pd.read_csv(read_path, header=None)
    df.columns = ['vidname'] + [f'frame_{i}' for i in range(seq_len)] + ['len', 'psnr', 'mse']
    ser = df.groupby('vidname').psnr.mean()
    df2 = pd.DataFrame(dict(vidname=ser.index, psnr=ser.values))
    df2.sort_values('psnr',inplace=True) # vidname,psnr
    df = df2[df2.psnr<=36]

    df['name'] = df['vidname'].map(lambda x:x[0] if len(x)<=2 else x[:-5])

    r2n = df.name.value_counts(sort=True, ascending=True).index
    n2r = {n:r for r,n in enumerate(r2n)}
    df['r_score'] = df['name'].map(lambda x: amp*n2r[x]/(len(r2n)-1))
    df['score'] = df['r_score']+df['psnr']

    df = df.sort_values('score')
    df.to_csv('diversity_considered.csv', index=False)

    dft = df.name.map(lambda x: x.find('Venice') != -1 or x.find('India') != -1)

    # visited = list()
    # mapping = dict()
    # for idx, k in enumerate(names):
    #     if k in visited: continue
    #     visited.append(idx)
    #     for i in range(4, 0, -1):
    #         if i >= len(k): continue
    #         # print(k)
    #         nk = k[:-i]
    #         print(k, nk)
    #         l = len(nk)
    #         for j in range(idx + 1, len(names)):
    #             if j in visited: continue
    #             if names[j][:l] == nk:
    #                 mapping[names[j]] = nk
    #                 mapping[k] = nk
    #                 visited.append(j)

    ind = df[dft].index
    dind = np.random.choice(ind, len(ind) // 2, replace=False)
    df = df.drop(index=dind)

    df2 = df
    # df2.to_csv('vid_rank.csv',index=False)
    # cumean = df2.expanding(1).mean()
    selected = df2[:230]
    print(selected.name.unique().shape)
    val_indices = np.random.choice(selected.index, 30, replace=False)
    train_indices = list(set(selected.index)-set(val_indices))
    val_set = df2.loc[val_indices]
    train_set = df2.loc[train_indices]
    val_set.to_csv(val_path,index=False)
    train_set.to_csv(train_path, index=False)
    print(f'trainset psnr {train_set.psnr.mean()}, valset psnr {val_set.psnr.mean()}')

if __name__ == '__main__':

    # a_fmt = r'/usr/whz/EDVSRGAN_root/VSR-DUF-master/inputs/G/{}/*.png'
    # b_fmt = r'/usr/whz/EDVSRGAN_root/VSR-DUF-master/results/52L/G/{}/*.png'
    # datasets = ['calendar','foliage','city','walk']
    # vsr.offline_performance([(a_fmt.format(x), b_fmt.format(x)) for x in datasets], datasets, sp_border=8)
    #######################3
    # a,b = parallel_rgb_ds_tf(a_fmt.format(datasets[0]),
    #                    b_fmt.format(datasets[0]),
    #                    'png')
    # it = [x.make_initializable_iterator() for x in [a,b]]
    # def graph_fn(it):
    #     a,b = [x.get_next() for x in it]
    #     # ops = [[tf.reduce_mean(x),tf.reduce_max(x),tf.reduce_min(x)] for x in [a,b]]
    #     a_,b_ = [image.ycbcr2rgb(image.rgb2ycbcr(x)) for x in [a,b]]
    #     ops = [tf.reduce_mean(tf.squared_difference(x,y)) for x,y in zip((a,b),(a_,b_))]
    #     return ops # list of 2 list
    #
    # def iter_fn(ops,sess):
    #     print(sess.run(ops))
    #
    # # ds_running_template(it, graph_fn, iter_fn)
    # a, b = [x.get_next() for x in it]
    # print(a.shape,b.shape)
    ########### generating data table
    data.scan('/usr/whz/vsr_data', None, '/usr/whz/EDVSRGAN_root/img_list.csv')
    vsr.detailize_data_file('/usr/whz/EDVSRGAN_root/img_list.csv', '/usr/whz/EDVSRGAN_root/img_detail_list.csv')

    ################ filtering dataset
    # fmt = '/usr/whz/EDVSRGAN_root/saved_models/pretrain_filter/{}.csv'
    # paths = ('test_results','valset','trainset')
    # paths = [fmt.format(x) for x in paths]
    # filter_ds(paths, 7)