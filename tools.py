from common_imports import *

def adjustible_gradient_clipping(optimizer, starter_lr, lr, loss, vars, max_norm):
    # TODO i just cancel the start_lr*
    adj_max_norm = starter_lr * max_norm / lr  # tf.reduce_max((self.starter_lr*10 / lr, 3.)) # 0.003/0.008
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr, name=name)
    # self.prompt('optimizer ' + name, 2)
    # ===== adjustable gradient clipping ==============
    grads_, vars = zip(*optimizer.compute_gradients(loss, var_list=vars))

    grads = []
    for idx, grad in enumerate(grads_):
        if grad is None:
            print('var={} with None grad'.format(vars[idx]))
            continue
        grads.append(tf.clip_by_norm(grad, tf.cast(adj_max_norm, tf.float32)))

    # if name == 'ADAM_pre':
    #     # tf.add_to_collection('grads', grads_[0])
    tf.add_to_collection('grads', ('og0',grads_[0]))
    tf.add_to_collection('grads', ('cg0',grads[0]))
    tf.add_to_collection('grads', ('og1',grads_[-1]))
    tf.add_to_collection('grads', ('cg1',grads[-1]))

    # o_grads0, c_grads0, o_grads1, c_grads1 = tf.get_collection('grads')

    return optimizer.apply_gradients(zip(grads_, vars))


def add_optimizer(optimizer, global_step, loss, ema_decay):
    '''Adds optimizer to the graph. Supposes that initialize function has already been called.
    '''
    with tf.variable_scope('optimizer'):
        # hp = self._hparams

        # Adam with constant learning rate
        # optimizer = tf.train.AdamOptimizer(learning_rate)#, hp.wavenet_adam_beta1,hp.wavenet_adam_beta2, hp.wavenet_adam_epsilon)

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients = gradients

        # Gradients clipping
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                      global_step=global_step)

    # Add exponential moving average
    # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    # Use adam optimization process as a dependency
    with tf.control_dependencies([adam_optimize]):
        # Create the shadow variables and add ops to maintain moving averages
        # Also updates moving averages after each update step
        # This is the optimize call instead of traditional adam_optimize one.
        # assert tuple(self.variables) == variables  # Verify all trainable variables are being averaged
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        optimize = ema.apply(variables)

    return optimize

def add_optimizer_gpu(optimizer,global_step, grads, ema_decay):
    '''Adds optimizer to the graph. Supposes that initialize function has already been called.
    '''
    with tf.variable_scope('optimizer'):
        # hp = self._hparams

        # Adam with constant learning rate
        # optimizer = tf.train.AdamOptimizer(learning_rate)#, hp.wavenet_adam_beta1,hp.wavenet_adam_beta2, hp.wavenet_adam_epsilon)

        # gradients = gradients
        gradients, variables = zip(*grads)
        # Gradients clipping
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                      global_step=global_step)

    # Add exponential moving average
    # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    # Use adam optimization process as a dependency
    with tf.control_dependencies([adam_optimize]):
        # Create the shadow variables and add ops to maintain moving averages
        # Also updates moving averages after each update step
        # This is the optimize call instead of traditional adam_optimize one.
        # assert tuple(self.variables) == variables  # Verify all trainable variables are being averaged
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        optimize = ema.apply(variables)

    return optimize

def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_shape(tensor):
    shape = tensor.get_shape()
    shape_list = shape.as_list()
    if not shape.is_fully_defined(): # find out which is not defined
        shape_arr = np.asarray(shape_list)
        indices = np.where(shape_arr==None)[0] # the indices are 1-d
        shape = tf.shape(tensor)
        for idx in indices:
            shape_list[idx] = shape[idx]

    return shape_list