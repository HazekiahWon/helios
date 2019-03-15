# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from tools import latest_ckpt_finder, add_custom_summ
import time
import traceback
import pandas as pd
from metrics import *

class Config(metaclass=ABCMeta):
    def __init__(self, batch_size, run_epoch, train_data_dir, val_data_dir,
                 save_iters_freq, summ_iters_freq,
                 starter_lr, max_norm, dir4allexp,
                 restore_exp, restore_ckpt):
        self.batch_size = batch_size
        self.run_epoch = run_epoch
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir

        self.save_iters_freq = save_iters_freq
        self.summ_iters_freq = summ_iters_freq

        self.starter_lr = starter_lr

        self.max_norm = max_norm

        self.dir4allexp = dir4allexp
        self.restore_exp = restore_exp
        self.restore_ckpt = restore_ckpt
        # self.seq_file = seq_file

    @abstractmethod
    def get_stage(self):
        pass
    @abstractmethod
    def get_memo(self):
        pass
    @abstractmethod
    def is_train(self):
        pass
    @abstractmethod
    def is_resume(self):
        pass


class Model(metaclass=ABCMeta):

    def __init__(self, model_name, config):

        self.model_name = model_name
        self.config = config
        self.stage = self.running_stage()
        self.is_train = config.is_train()

        self.is_resume = config.is_resume() #config.resume if self.running_type == 'train' else True  # val and test all use resume

        self.batch_size = config.batch_size

        self.run_epoch = config.run_epoch

        self.train_data_dir = config.train_data_dir
        self.test_data_dir = config.val_data_dir

        self.save_freq = config.save_iters_freq
        self.summ_freq = config.summ_iters_freq

        self.starter_lr = config.starter_lr

        self.max_norm = config.max_norm
        # self.graph = tf.Graph()
        self.sess = None  # prepare for dynamic eval()
        self.saver = None

        self.logger = logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.getHandler(logging.StreamHandler(),
                                         '%(asctime)s - %(message)s',
                                         logging.WARNING))

        start_time = time.strftime('%H%M%S_%Y%m%d', time.localtime())

        self.save_ckpts_dirpath = os.path.join(config.dir4allexp,
                                               '{}_'.format(self.stage) + start_time)
        os.makedirs(self.save_ckpts_dirpath, exist_ok=True)

        self.logger.addHandler(self.getHandler(logging.FileHandler(f'{self.save_ckpts_dirpath}/logger.txt'),
                                               '%(asctime)s - %(message)s',
                                               logging.INFO))
        self.prompt(config.get_memo())

        # self.seq_table_path = os.path.join(self.save_ckpts_dirpath, config.seq_file)
        self.train_report_path = os.path.join(self.save_ckpts_dirpath, 'train_results.csv')
        self.eval_report_path = os.path.join(self.save_ckpts_dirpath, 'eval_results.csv')

    def running_stage(self):
        t = self.config.get_stage()
        if isinstance(t, tuple):
            return '_'.join(t)
        else: return t

    @staticmethod
    def getHandler(handler, format, level):
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        return handler

    def prompt(self, s, level=0):
        if level==0: self.logger.info(s)
        # elif level==1: self.logger.info(s)
        else: self.logger.warning(s)


    def save_settings(self):
        attrs = self.__dict__
        lr_type = type(self.starter_lr)
        attrs = {k:v for k, v in attrs.items() if type(v) in (str, int, lr_type, bool, dict)}
        prompts = ['='*25+'hyperparams'+'='*25,
                   '\n'.join(f'{k}={v}' for k,v in attrs.items()),
                   '=' * 25 + 'hyperparams' + '=' * 25]
        self.prompt('\n'.join(prompts))
        np.save(os.path.join(self.save_ckpts_dirpath, 'attrs.npy'), attrs)

    # during model build
    @abstractmethod
    def _inputs_preprocess(self, inputs):
        pass
    @abstractmethod
    def _postprocess_outputs(self, predicted_raw, uv):
        pass

    def average_gradients_across_towers(self, tower_grads, max_norm=3.):
        """Calculate average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been
           averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars: # same var for different tower
                if g is None: print(_)

                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        if max_norm is not None:
            grads, _ = tf.clip_by_global_norm([x[0] for x in average_grads], max_norm)
            average_grads = list(zip(grads,[x[1] for x in average_grads]))
        return average_grads

    @abstractmethod
    def _build_model(self):  # no attention
        pass
        # inputs = iterator.get_next()
        #
        # pre_ret, metric_operand_hr, loss_operand_hr, uv = self._inputs_preprocess(to_preprocess)
        #
        # predicted_raw = models.lsvsr(inputs, **self.g_net_configs)
        # predicted_v, metric_operand_y, loss_operand_y = self._postprocess_outputs(predicted_raw, uv)
        #
        # ret = self._build_loss((metric_operand_hr, metric_operand_y, loss_operand_hr, loss_operand_y), loss_mask,
        #                            average_psnr=average_psnr)
        #
        # return vidn, predicted_v,ret # ret: first is loss, the rest is for record

    @abstractmethod
    def _build_model_multi(self, iterator):  # no attention
        pass
        # self.lr_g = tf.get_variable('lr_g', initializer=self.starter_lr, trainable=False)
        #
        # g_optimizer = tf.train.AdamOptimizer(self.lr_g)
        #
        # tower_grads = []
        # tmp = list()
        # with tf.variable_scope(tf.get_variable_scope()) as scope:
        #     for d in get_available_gpus():
        #         with tf.device(d):
        #             with tf.name_scope(f'tower_{d[-1]}'):
        #                 _,predicted_v,ret = self._build_model(iterator, reuse=len(tmp) != 0, is_train_call=True)
        #
        #                 tmp.append(ret)
        #                 loss = ret[0]
        #
        #                 grad_var_pairs = g_optimizer.compute_gradients(loss, var_list=self.var_list)
        #                 tower_grads.append(grad_var_pairs)
        #         scope.reuse_variables()
        #
        # ret = list()
        # tmp = list(zip(*tmp)) # list of two-element tuple
        # losses = tmp[0] # a list of loss
        # tmp = tmp[1:]
        # for e in tmp:  # e is a list of 3
        #     ret.append(tf.add_n(e) / len(e))
        #
        # self.losses = dict(g_loss=tf.add_n(losses)/len(losses))
        #
        # avg_grad = self.average_gradients_across_towers(tower_grads, max_norm=self.max_norm)
        # self.trainer = g_optimizer.apply_gradients(avg_grad)
        #
        # return predicted_v,ret # psnr,ssim

    @abstractmethod
    def _build_loss(self):
        pass

    @abstractmethod
    def _set_optim_vars(self):
        pass

    @abstractmethod
    def _build_summaries(self):  # only used in training
        pass

    def _params_usage(self):
        total = 0
        prompt = []
        for v in tf.trainable_variables():
            shape = v.get_shape()
            cnt = 1
            for dim in shape:
                cnt *= dim.value
            prompt.append('{} with shape {} has {}'.format(v.name, shape, cnt))
            # self.prompt(prompt[-1])
            total += cnt
        prompt.append('totaling {}'.format(total))

        return '\n'.join(prompt)

    def resume(self, var_list, resume_exp_dir=''):

        restorer = tf.run.Saver(var_list)

        if not resume_exp_dir:
            resume_exp_dir = latest_ckpt_finder(self.config.dir4allexp, need_expand_myself=True)
        if isinstance(resume_exp_dir, tuple):
            exp_dir, model_name = resume_exp_dir
            model_name = os.path.join(self.config.dir4allexp, exp_dir, model_name)
        else:
            exp_dir = resume_exp_dir
            model_name = tf.run.latest_checkpoint(os.path.join(self.config.dir4allexp, exp_dir))
        restorer.restore(self.sess, model_name)
        self.prompt("Restored from {}".format(model_name), 2)


    def save(self, prefix, step):
        self.prompt("Saving...")
        self.saver.save(self.sess, os.path.join(self.save_ckpts_dirpath, prefix), global_step=step)
        self.prompt("Saved!")
    # @abstractmethod
    def construct_dataset(self, iter_fn, input_format, shapes, types, pad_shapes, pad_values, batch_size=1):
        # types = dict(fname=tf.string, lr=tf.float32, hr=tf.float32, len=tf.int32)
        # shapes = dict(fname=(), lr=(None, *lr_size, 3), hr=(None, *hr_size, 3), cond=(self.cond_len, *hr_size, 3), len=())
        # pad_shapes = dict(shapes, lr=(self.max_time, *lr_size, 3), hr=(self.max_time, *hr_size, 3))
        # pads = dict(fname='', lr=0., hr=0., len=0)
        # input_format = ['fname','hr','len','lr']
        ds = tf.data.Dataset.from_generator(iter_fn,
                                            output_shapes=tuple([shapes[k] for k in input_format]),  #((),(None,*in_size,3),(),(None,*out_size)),#
                                            output_types=tuple([types[k] for k in input_format]))
        ds = ds.padded_batch(batch_size,
                             padded_shapes=tuple([pad_shapes[k] for k in input_format]),
                             padding_values=tuple([pad_values[k] for k in input_format]),
                             drop_remainder=True)

        return ds.make_initializable_iterator()

    def validate_save(self, epoch, mean, psnrs):

        convertor = lambda x, y: (x * len(y) - 1) / (len(y) - 1) * 100
        # self.prompt('=========validation========')
        df = pd.DataFrame(np.array([epoch]+psnrs).reshape((1,-1)))
        df.to_csv(os.path.join(self.save_ckpts_dirpath, 'val_results.csv'), mode='a', header=False, index=False)

        add_custom_summ(self.writer, 'vid4', mean, epoch)
        # results.append(cal)
        rec = self.records  # [dataset]
        rec.append(mean)
        # thresh = np.inf
        if epoch > 20:
            del rec[0]
        if epoch > 5:
            thresh = np.percentile(rec, convertor(0.7, rec))
        else: thresh = 0

        prefix = None
        if mean > thresh:
            prefix = str(mean).replace('.', '_')
        elif (epoch + 1) % self.save_freq == 0:
            prefix = 'train'

        if prefix is not None:
            self.save(prefix=prefix, step=epoch)
            self.prompt('saved as {}'.format(prefix),0)

        return mean

    def debug_ds(self, n_iters):
        # iterator = self.construct_dataset(os.path.join(self.train_data_dir, config.sub_data_dir),
        #                                        file_suffix='bmp', n_samples=n_samples)
        # self.sess.run(iterator.initializer)
        # cnt = 0
        #
        # while True:
        #     try:
        #         lr,hr,length,cond = iterator.get_next()
        #         hr_ = rgb2ycbcr(hr)
        #         hr,hr_ = self.sess.run([hr,hr_])
        #
        #         print(pd.Series(hr.flatten()).describe())
        #         print(pd.Series(hr_.flatten()).describe())
        #         print(pd.Series(hr_[:,:,:,:,0].flatten()).describe())
        #         input()
        #         cnt += 1
        #     except tf.errors.OutOfRangeError:
        #         print('============',cnt)
        #         break
        ################################################################333
        # self.before_run()
        #
        # print('init vars ...')
        #
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(self.iterator.initializer)
        #
        # while True:
        #     try:
        #         vars = self.sess.run(self.debugs)
        #         for idx,v in enumerate(vars):
        #             print('=====', idx)
        #             print(pd.Series(v.flatten()).describe())
        #         input()
        #     except tf.errors.OutOfRangeError:
        #         break
        #############################################################
        # g = tf.Graph()
        seed = 233333
        self.graph = tf.get_default_graph()
        conf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        conf.gpu_options.allow_growth = False
        for _ in range(2):
            with tf.Session(config=conf) as s:
                self.sess = s
                self.set_seed(seed)
                # iterator = self.construct_dataset(os.path.join(self.train_data_dir, config.sub_data_dir),
                #                                        real_size=(480, 640), out_size=(self.out_h, self.out_w), n_samples=50)  # 50 folder out of 510
                iterator = self.construct_dataset(self.train_data_dir, lr_size=(self.in_h, self.in_w), n_iters=n_iters)  # 50 folder out of 510

                # with tf.Session(graph=) as sess:
                self.sess.run(iterator.initializer)
                while True:
                    try:
                        st = time.time()
                        vars = self.sess.run(iterator.get_next())
                        dur = time.time()-st
                        print(vars[0], dur)
                    except tf.errors.OutOfRangeError:
                        break

    def set_seed(self, reproduce=None):
        if reproduce is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
            self.prompt('seed {}'.format(seed))
        else:
            seed = reproduce
            self.prompt('reproducing with seed {}'.format(reproduce))

        np.random.seed(seed)
        tf.set_random_seed(seed)

    @abstractmethod
    def prepare_val(self):
        pass
        # self.val_iterator = self.construct_dataset(self.test_data_dir, lr_size=(None, None), batch_size=1)  # 50 folder out of 510
        # self.val_vidn, _, r = self._build_model(self.val_iterator, reuse=True, is_train_call=False, average_psnr=False)
        # self.val_metrics = Metrics(r[1:])
    @abstractmethod
    def prepare_train_mode(self):
        pass
        # self.set_seed(seed)
        # self.save_settings()
        # self.iterator = self.construct_dataset(config.train_data_dir, lr_size=(self.in_h, self.in_h),
        #                                        n_iters=config.n_iters_per_epoch if not data_use_all else None)  # either use all or use config
        #
        # self.predicted_videos, r = self._build_model_multi(self.iterator)
        # self.train_metrics = Metrics(r)
        # self._build_summaries()

    def prepare_saver_writer(self):
        self.prompt(self._params_usage())

        self.prompt('init saver,writer ...')
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)  #

        if self.save_ckpts_dirpath:
            self.writer = tf.summary.FileWriter(self.save_ckpts_dirpath,
                                                graph=tf.get_default_graph())  # tf.get_default_graph()
        else:
            self.writer = None
        self.prompt('finish establishing.', 2)
        self.prompt('=' * 53, 2)


    def train_template(self, iterator, bo_epoch, before_run, prepare_nodes, after_run, after_epoch):
        iters = 0
        # bo_epoch = self.run_epoch
        result = 0
        ret = None
        for ep in range(bo_epoch):
            if result==-1: break # oom
            self.prompt('='*10+" Epoch {}/{}".format(ep, bo_epoch),2)

            # epoch_elapsed = 0.
            self.sess.run(iterator.initializer)
            init = before_run()
            while True: # iter thru an epoch
                try:
                    nodes = prepare_nodes(iters)
                    start = time.time()
                    sess_ret = self.sess.run(nodes)
                    elapsed = time.time() - start
                    after_run(iters, init, sess_ret, elapsed)
                    iters += 1

                except tf.errors.OutOfRangeError:
                    break # into next epoch
                except Exception as e:
                    logging.error(repr(e))
                    traceback.print_exc()
                    result = -1
                    break
            ret = after_epoch(init, iters)
            self.prompt('=' * 10 + " Epoch {}/{} finishes".format(ep, bo_epoch), 2)
        return ret

    def run(self, resume_model_dir='', vl=None):

        with self.graph.as_default(), tf.device("/cpu:0"):
            configpro = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            configpro.gpu_options.allow_growth = True
            with tf.Session(config=configpro) as s:  # Session use the default graph, exactly the self.graph
                self.sess = s
                #??
                self.prepare_train_mode()
                self.prepare_val()
                self.prepare_saver_writer()

                print('init vars ...')
                self.sess.run(tf.global_variables_initializer())

                if self.config.need_resumed_model():
                    # can be resuming dt, or not using dt
                    if not resume_model_dir:  # not explicit in param
                        if not self.config.restore_ckpt:  # has not config of ckpt
                            if self.config.restore_exp: resume_model_dir = self.config.restore_exp  # has config of exp
                        else:
                            resume_model_dir = (self.config.restore_exp, self.config.restore_ckpt)  # has ckpt means has exp
                    if vl is None:
                        vl = [x for x in self.var_list if x.name.startswith(self.G_scope_name)]
                    self.resume(vl, resume_model_dir)  # when dt, it is all vars, otherwise only g vars
                # ??
                self.train_template()

                print('Finish Training')


if __name__ == '__main__':
    pass