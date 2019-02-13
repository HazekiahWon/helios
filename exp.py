from common_imports import *
import image
# from . import vsr
# from . import tools

def parallel_name_ds_tf(src_a, src_b, seed=0):
    a = tf.data.Dataset.list_files(src_a, seed=seed)
    b = tf.data.Dataset.list_files(src_b, seed=seed)
    # ds = tf.data.Dataset.zip((a,b))
    # it = ds.make_initializable_iterator()
    return a,b

def parallel_img_ds_tf(src_a, src_b, format, map_fn, seed=0, batch=None, keep_names=False):
    ds_a_,ds_b_ = parallel_name_ds_tf(src_a, src_b, seed)

    ds_a = ds_a_.map(map_fn)
    ds_b = ds_b_.map(map_fn)
    if batch is not None:
        ds_a = ds_a.batch(batch)
        ds_b = ds_b.batch(batch)

    if keep_names:
        name_it = ds_a_.make_initializable_iterator()
        return ds_a,ds_b,name_it
    else: return ds_a,ds_b

def parallel_rgb_ds_tf(src_a, src_b, format, seed=0, batch=None, keep_names=False):
    def map_fn(x):
        rgb = image.read_img_tf(x, format)
        rgb = image.uint8_to_float(rgb,tf.float32)

        return rgb

    return parallel_img_ds_tf(src_a, src_b, format, map_fn, seed, batch, keep_names)

def parallel_ycbcr_ds_tf(src_a, src_b, format, seed=0, batch=None, keep_names=False, sp_border=0):
    def map_fn(x):
        rgb = image.read_img_tf(x, format)
        rgb = image.uint8_to_float(rgb, tf.float32)
        ycbcr = image.rgb2ycbcr(rgb)
        # [:, :, sp_border:h - sp_border, sp_border:w - sp_border, :]
        if ycbcr.shape.ndims==4:

            return ycbcr[:,sp_border:- sp_border, sp_border:- sp_border,0:1]
        elif ycbcr.shape.ndims==3:
            # h, w, _ = ycbcr.get_shape().as_list()
            return ycbcr[sp_border:- sp_border, sp_border:- sp_border,0:1]
        else: raise Exception(f'wrong dimension {ycbcr.shape.ndims} for images')

    return parallel_img_ds_tf(src_a, src_b, format, map_fn, seed, batch, keep_names)

def parallel_psnr_compare_tf(src_a, src_b, img_format, compare_type='ycbcr', seed=0, batch=None, keep_names=False, do_average=False, sp_border=0):
    if compare_type=='rgb':
        ret = parallel_rgb_ds_tf(src_a, src_b, img_format, seed, batch, keep_names)
    else: # ycbcr
        ret = parallel_ycbcr_ds_tf(src_a, src_b, img_format, seed, batch, keep_names, sp_border)

    if keep_names:
        a,b,name = ret
    else: a,b = ret
    a_it = a.make_initializable_iterator()
    b_it = b.make_initializable_iterator()
    a_nex = a_it.get_next()
    b_nex = b_it.get_next()
    psnr = image.batch_image_psnr_tf(a_nex,b_nex, do_average=do_average)
    if keep_names:
        name_it = name.make_initializable_iterator()
        return {'it':(a_it,b_it,name_it), 'op':(psnr, name_it.get_next())}
    else: return {'it':(a_it,b_it), 'op':psnr}

def ds_running_template(it, graph_fn, iter_fn, finish_fn=None):
    with tf.Session() as sess:
        if not isinstance(it, list): sess.run(it.initializer)
        else: [sess.run(i.initializer) for i in it]
        ret = graph_fn(it)
        returns = list()
        while True:
            try:
                returns.append(iter_fn(ret,sess))
            except tf.errors.OutOfRangeError:
                break
        if finish_fn is not None:
            finish_fn(returns)

def stats(arr):
    return np.min(arr), np.mean(arr), np.max(arr)


