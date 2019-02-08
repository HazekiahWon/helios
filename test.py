import sys
print(sys.path)
import data,vsr
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
    data.scan('/usr/whz/vsr_data', None, '/usr/whz/EDVSRGAN_root/img_list.csv')
    vsr.detailize_data_file('/usr/whz/EDVSRGAN_root/img_list.csv', '/usr/whz/EDVSRGAN_root/img_detail_list.csv')