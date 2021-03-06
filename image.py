from common_imports import *
# from PIL import Image
from scipy.misc import toimage,fromimage,imread,imsave
from scipy import signal
from tools import get_shape
# from scipy.ndimage import zoom
ycbcr_from_rgb = np.array([[    65.481,   128.553,    24.966],
                           [   -37.797,   -74.203,   112.0  ],
                           [   112.0  ,   -93.786,   -18.214]])/255.

rgb_from_ycbcr = np.linalg.inv(ycbcr_from_rgb)

origOffset = np.array([16.0, 128.0, 128.0]) # (3,)
def extract_image_patches(x, k):
    b,H,W,_ = get_shape(x)
    p = tf.extract_image_patches(x,
                                 ksizes=[1, k, k, 1], strides=[1, k, k, 1], rates=[1, 1, 1, 1], padding='SAME')
    b, h, w, c = get_shape(p)
    p = tf.reshape(p, (b*h*w, k, k, c // (k ** 2)))
    return p

def rgb2ycbcr(inputs):
    """

    :param inputs: rgb in [0,255.]
    :return: ycbcr in [16,235]
    """
    with tf.name_scope('rgb2ycbcr'):
        # the following requires float32
        if inputs.dtype!=tf.float32: inputs = tf.cast(inputs, tf.float32)

        shape = inputs.get_shape()
        # gray image
        if shape[-1].value == 1:
            return inputs
        assert shape[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'

        w = tf.constant(ycbcr_from_rgb,dtype=tf.float32) # (3,3)
        # 1,1,1,3,3

        x = tf.stack([inputs]*3,axis=-2) # b,h,w,3,3
        output = tf.reduce_sum(w*x, axis=-1)+origOffset # b,h,w,3
        # n_ones = ndims-1
        # w = tf.reshape(origT, [1]*n_ones+[3,3])/255. # 4: 1,1,1,3,3
        # x = tf.expand_dims(inputs, axis=ndims) # b,h,w,3,1
        # w = tf.tile(w, multiples=shape.as_list()[:n_ones]+[1,1])
        # output = tf.squeeze(tf.matmul(w,x))+origOffset  # b,h,w,3
        return output

def ycbcr2rgb(inputs): # y = wx+b  x = w' y-b (3,3) (1,3)
    """

    :param inputs: in [0,255.], the function will 255* for you (line70)
    :return: rgb in [0,255]
    """
    with tf.name_scope('ycbcr2rgb'):
        shape = inputs.get_shape()
        assert shape[-1].value == 3, 'Error: ycbcr2rgb input should be RGB or grayscale!'

        w = tf.constant(rgb_from_ycbcr,dtype=tf.float32)  # (3,3)
        # 1,1,1,3,3
        # origOffset = np.array([16.0, 128.0, 128.0])  # (3,)
        x = tf.stack([inputs-origOffset] * 3, axis=-2)  # b,h,w,3,3
        output = tf.reduce_sum(w * x, axis=-1)  # b,h,w,3
        return output


def get_image_decoder_tf(format):
    if format=='bmp': return tf.image.decode_bmp
    elif format=='png': return tf.image.decode_png
    else: raise Exception(f'Unknown image format {format}')

def read_img_tf(fname_tensor, format):
    decoder = get_image_decoder_tf(format)
    return decoder(tf.read_file(fname_tensor), channels=3)

def batch_image_psnr_tf(a, b, clip_max=100., clip_min=0., do_average=False):
    """

    :param a: 4-d tensor, b,h,w,c
    :param b: 4-d tensor
    :param clip_max:
    :param clip_min:
    :param do_average:
    :return: (b,),[(,)]
    """
    batch_psnr = tf.clip_by_value(tf.image.psnr(a, b, max_val=255), clip_min, clip_max)
    if do_average:
        avg_psnr = tf.reduce_mean(batch_psnr)
        return batch_psnr,avg_psnr
    else: return batch_psnr

def uint8_to_float(tensor, dtype):
    """

    :param tensor: single image
    :param dtype:
    :return:
    """
    return tf.image.convert_image_dtype(tensor,dtype)


def resize_image(arr, size, interp='bicubic', set_int=True, save_path=None):
    """
    Resize an image.
    This function is only available if Python Imaging Library (PIL) is installed.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Parameters
    ----------
    arr : ndarray
        The array of image to be resized. (*single image)
    size : int, float or tuple
        [not allowed] * int   - Percentage of current size.
        [not allowed] * float - Fraction of current size.
        * tuple - Size of the output image (height, width).
    interp : str, optional
        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
        'bicubic' or 'cubic').
    set_int : set the returned nparray uint
    [* depreated]
    mode : str, optional
        The PIL image mode ('P', 'L', etc.) to convert `arr` before resizing.
        If ``mode=None`` (the default), 2-D images will be treated like
        ``mode='L'``, i.e. casting to long integer.  For 3-D and 4-D arrays,
        `mode` will be set to ``'RGB'`` and ``'RGBA'`` respectively.
    Returns
    -------
    imresize : ndarray
        The resized array of image. in range (0,255), dtype is uint32 if set_int, otherwise float32
    See Also
    --------
    toimage : Implicitly used to convert `arr` according to `mode`.
    scipy.ndimage.zoom : More generic implementation that does not use PIL.
    --------
    Reference: https://github.com/scipy/scipy/blob/maintenance/1.2.x/scipy/misc/pilutil.py
    """
    if save_path is not None and not set_int:
            logging.warning('set_int is set True when saving images')
            set_int = True

    mode = 'F' if not set_int and len(arr.shape)==2 else None
    im = toimage(arr, mode=mode) # PIL.Image

    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size[::-1], resample=func[interp]) # PIL.Image.resize

    if save_path is not None:
        imnew.save(save_path)
    return fromimage(imnew)

# def batch_image_resize(arr, size, interp='bicubic', set_int=True):
#     """
#
#     :param arr: b,hwc
#     :param size:
#     :param interp:
#     :param set_int:
#     :return:
#     """
#     func = {'nearest': 0, 'bilinear': 1, 'bicubic': 3, 'cubic': 3}
#     dtype = np.int32 if set_int else np.float32
#     return zoom(arr, [1]+list(size)+[1], output=dtype, order=func[interp])

def read_image(path):
    return imread(path)

def save_image(path, im):
    imsave(path, im)

def crop_offsets(in_size, out_size):
    """

    :param in_size:
    :param out_size:
    :return: single offsets, a:b,c:d
    """
    a,c = [np.random.randint(0, x-y) for x,y in zip(in_size,out_size)]
    b,d = [x+y for x,y in zip((a,c),out_size)]

    return a,b,c,d


def gauss_kernel2d(ksize, stddev=None):
    """

    :param ksize:
    :param stddev:
    :return:
    plot
    -------
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(gauss_kernel2d(21), interpolation='none')
    Reference
    -------
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    if stddev is None: stddev = (ksize - 1)   # 3sigma for normal distr.
    gkern1d = signal.gaussian(ksize, std=stddev).reshape(ksize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def gauss_deconv2d(inp, ksize, stride, name):
    # 不失真的话ksize=stride
    # ksize=stride*reception, reception is unit length of an object
    b,h,w,c = get_shape(inp)
    k = gauss_kernel2d(ksize).astype(np.float32).reshape((ksize,ksize,1,1))
    c_out = c_in = c
    k = np.tile(k, (1, 1, c_out, c_in))
    # print(k.shape)
    # k = tf.get_variable(name, initializer=k)
    return tf.layers.conv2d_transpose(inp, c, ksize, stride, padding='same', kernel_initializer=tf.constant_initializer(k))
    return tf.nn.conv2d_transpose(inp, k,
                                  [b,h*stride,w*stride,c], [1,stride,stride,1], padding='SAME')

def gauss_deconv3d(inp, ksize, stride):
    # 不失真的话ksize=stride
    # ksize=stride*reception, reception is unit length of an object
    b,t,h,w,c = get_shape(inp)
    k = gauss_kernel2d(ksize).astype(np.float32).reshape((1,ksize,ksize,1,1)) # t,h,w,out,in
    c_out = c_in = c
    k = tf.tile(k, (1, 1, 1, c_out, c_in))
    return tf.nn.conv3d_transpose(inp, k,
                                  [b,t,h*stride,w*stride,c], [1,1,stride,stride,1], padding='SAME')

