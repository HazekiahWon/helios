from pipeline import *
from tools import get_shape, floor_by
from image import read_image, rgb2ycbcr, resize_image
# from exp import stats
class FTConfig(Config):
    def __init__(self):
        super().__init__(
            batch_size=4,
            run_epoch=50,
            train_data_dir=r'D:\helios\facetype\脸型',
            val_data_dir=r'D:\helios\facetype\脸型',
            save_iters_freq=None,
            summ_iters_freq=25,
            starter_lr=1e-3,
            max_norm=10.,
            dir4allexp=r'D:\helios\facetype\exp',
            restore_exp=None,
            restore_ckpt=None
        )
        self.major_stage = 0
        self.sub_stage = 0
        self.memo = '1'
        self.n_classes = 7
        self.h,self.w = 240,360

    def get_stage(self):
        s = ['train', 'val', 'test']
        ts = ['normal', 'resume', 'finetune']
        if self.major_stage == 0 and self.sub_stage > 0:
            return s[self.major_stage], ts[self.sub_stage]
        else:
            return s[self.major_stage]

    def need_resumed_model(self):
        return not (self.is_train() and self.sub_stage == 0)

    def get_memo(self):
        m = self.memo + ': {}'

        if self.major_stage == 0 and self.sub_stage != 0:
            x = ' '.join(self.get_stage()) + f' {restore_exp}'
        else:
            x = self.get_stage()

        return m.format(x)

    def is_train(self):
        return self.major_stage==0

    def is_resume(self):
        return self.major_stage==0 and self.sub_stage==1

class Prototype:
    @staticmethod
    def cnn(inp, is_train, filters, n_classes):
        # inp, labels = iterator.get_next()
        # filters = [(32, 5), (64, 5), 1024]
        # 240,360 - 120,180 - 60, 90 - 30,45 - 15,22
        cols = dict()
        cols['images'] = list()
        cnt = 0
        for nf, k in filters[:-1]:
            hold = inp
            inp = tf.layers.conv2d(
                inputs=inp,
                filters=nf,
                kernel_size=k,
                padding="same",
                activation=tf.nn.leaky_relu,
                name=f'fmap_{cnt}')
            inp = inp+tf.layers.conv2d(hold,nf,1, name=f'skip_linear_{cnt}')
            cnt+=1
            cols['images'].append(inp[:,:,:,0:1])
            if cnt%2==0:
                inp = tf.layers.max_pooling2d(inputs=inp, pool_size=[2, 2], strides=2)

        b, h, w, c = get_shape(inp)
        dense = tf.layers.conv2d(inp, filters[-1], kernel_size=(h,w), padding='valid') # 1,1,c
        dense = tf.reshape(dense, (b,-1))
        # flat = tf.reshape(inp, [b, h * w * c])
        # dense = tf.layers.dense(inputs=flat, units=filters[-1], activation=tf.nn.relu)
        # dropout = tf.layers.dropout(
        #     inputs=dense, rate=0.4, training=is_train)
        dropout = dense
        logits = tf.layers.dense(inputs=dropout, units=n_classes, name='logits')
        return logits,cols

class FT(Model):
    def __init__(self, config):
        super().__init__('facetype', config)
        self.data_file = r'D:\helios\facetype\trainset.csv'
        self.eval_data_file = r'D:\helios\facetype\valset.csv'
        self.n_classes = config.n_classes
        self.net_configs = dict(filters=[(32,3)]*10+[64], n_classes=self.n_classes)
        self.h,self.w = config.h,config.w

    def construct_dataset(self, batchsize, data_file):
        # iter_fn
        def iter_fn():
            df = pd.read_csv(data_file, usecols=['path','class']).values
            np.random.shuffle(df)
            df = df[:floor_by(df.shape[0], batchsize)]
            self.xtable = df
            # force the readed image to be 3 in dim3
            for p,c in df:
                im = read_image(p)
                if im.ndim<3: im = np.stack(3*(im,),axis=-1)
                elif im.shape[-1]>3: im = im[:,:,:3]
                im = resize_image(im, (self.h,self.w), set_int=False)

                yield p,im.astype(np.float32),c

        # input_format, types shapes, etc
        input_format = ['path','img','cls']
        types = dict(path=tf.string, img=tf.float32,cls=tf.int32)
        shapes = dict(path=(), img=(self.h,self.w,3),cls=())
        pad_shapes = dict(path=(), img=(self.h,self.w,3), cls=())
        pad_values = dict(path='', img=0., cls=0)
        iterator = super().construct_dataset(iter_fn, input_format, shapes,
                                  types, pad_shapes, pad_values, batch_size=batchsize)
        return iterator

    def debug_ds(self):
        seed = 233333
        # self.graph = tf.get_default_graph()
        conf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        conf.gpu_options.allow_growth = False
        for _ in range(1):
            with tf.Session(config=conf) as s:
                self.sess = s
                self.set_seed(seed)
                iterator = self.construct_dataset(batchsize=1, data_file=self.data_file)  # 50 folder out of 510
                self.sess.run(iterator.initializer)
                cnt = 0
                shape_col = list()
                while True:
                    try:
                        st = time.time()
                        vars = self.sess.run(iterator.get_next())
                        dur = time.time()-st
                        shape_col.append(vars[1].shape)
                        cnt += 1
                    except tf.errors.OutOfRangeError:
                        print(f'totaling {cnt}')
                        break
                shape_col = np.stack(shape_col)

                print(pd.DataFrame(shape_col).describe())

    def _build_model(self, iterator, is_train):
        _, inp, labels = iterator.get_next()
        inp = self._inputs_preprocess(inp)
        logits,cols = Prototype.cnn(inp, is_train, **self.net_configs)
        probas = tf.nn.softmax(logits, name='probas')
        if is_train:
            tf.add_to_collection('hist',logits)
            tf.add_to_collection('hist',probas)
        if is_train:
            for k,v in cols.items():
                [tf.add_to_collection(k,x) for x in v]
        xent = self._build_loss(logits, labels)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32) # b,n_classes
        # print(logits.shape, predictions.shape)
        # falsehood = tf.clip_by_value(tf.abs(labels-predictions),0,1) # 1 means false
        # accuracy =
        return xent, predictions

    def _inputs_preprocess(self, inputs):
        return rgb2ycbcr(inputs)

    def _postprocess_outputs(self, outs):
        pass

    def _build_model_multi(self, iterator, is_train):
        # optimizer
        with tf.variable_scope('model', reuse=not is_train):
            loss, pred = self._build_model(iterator, is_train)
        if is_train:
            with tf.variable_scope('optimize'):
                self.lr = tf.get_variable('lr', initializer=self.starter_lr, trainable=False)
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.trainer = optimizer.minimize(loss)
            return loss, pred
        else: return pred

    def _build_loss(self, logits, labels):
        xent = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        # tf.add_to_collection('hist',logits)
        xent = tf.reduce_mean(xent, name='xent_loss') # **xent has been averaged
        tf.add_to_collection('scalar', xent)
        return xent

    def _set_optim_vars(self):
        self.var_list = tf.trainable_variables()

    def _build_summaries(self):
        scalars = [tf.summary.scalar(x.op.name,x) for x in tf.get_collection('scalar')]
        imgs = [tf.summary.image(x.op.name, x) for x in tf.get_collection('images')]
        hists = [tf.summary.histogram(x.op.name, x) for x in tf.get_collection('hist')]
        all = scalars+imgs+hists
        self.summ = tf.summary.merge(all)

    def prepare_val(self):
        self.val_iterator = self.construct_dataset(self.batch_size, self.eval_data_file)
        self.val_pred = self._build_model_multi(self.val_iterator, is_train=False)

    def prepare_train_mode(self, seed=None):
        self.set_seed(seed)
        self.save_settings()
        self.train_iterator = self.construct_dataset(self.batch_size, self.data_file)
        self.xent, self.pred = self._build_model_multi(self.train_iterator, is_train=True)
        self._build_summaries()

    def train(self):
        def before_run():
            return list()
        def prepare_nodes(iters):
            return [self.trainer, self.pred, self.xent, self.summ]
        def after_run(iters, init, sess_ret, elapsed):
            _, b_pred,m_xent, summ = sess_ret
            self.writer.add_summary(summ, global_step=iters)

            pred_col = init
            pred_col.append(b_pred)
            # false_col.append(b_false)
            self.prompt(f'iter {iters} elapsed {elapsed} with xent {m_xent}.',2)

            return pred_col
        def after_epoch(init, iters):
            pred_col= init
            # rlen = len(self.xtable)
            # try to make sure they are of same length
            pred = np.concatenate(pred_col).reshape((-1,1))
            # false = np.concatenate(false_col)
            # acc = 1-np.mean(false)

            # print(self.xtable.shape, pred.shape)
            xytable = np.concatenate((self.xtable,pred), axis=1)
            acc = np.mean(xytable[:,-1]==xytable[:,-2])
            self.prompt(f'the overall acc is {acc}.', 2)
            pd.DataFrame(xytable, columns=['path','class','pred']).to_csv(self.train_report_path, index=False)
            acc = self.evaluate()
            self.save(str(acc)[:5], iters)
        self.train_template(self.train_iterator, self.run_epoch, before_run, prepare_nodes, after_run, after_epoch)

    def evaluate(self):
        def before_run():
            return list()
        def prepare_nodes(iters):
            return self.pred
        def after_run(iters, init, sess_ret, elapsed):
            b_pred = sess_ret # make sure batchsize only 1
            pred_col = init
            pred_col.append(b_pred)
            # false_col.append(b_false)
            self.prompt(f'iter {iters} elapsed {elapsed} with xent {m_xent}.',2)

            return pred_col
        def after_epoch(init, iters):
            pred_col= init
            # rlen = len(self.xtable)
            # try to make sure they are of same length
            pred = np.concatenate(pred_col).reshape((-1,1))
            # false = np.concatenate(false_col)
            # acc = 1-np.mean(false)

            # print(self.xtable.shape, pred.shape)
            xytable = np.concatenate((self.xtable,pred), axis=1)
            acc = np.mean(xytable[:,-1]==xytable[:,-2])
            self.prompt(f'the overall acc is {acc}.', 2)
            pd.DataFrame(xytable, columns=['path','class','pred']).to_csv(self.eval_report_path, index=False)
            return acc

        return self.train_template(self.val_iterator, 1, before_run, prepare_nodes, after_run, after_epoch)

    def run(self, resume_model_dir='', vl=None):
        with tf.device("/cpu:0"):
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
                        vl = [x for x in self.var_list if x.name.startswith(self.model_name)]
                    self.resume(vl, resume_model_dir)  # when dt, it is all vars, otherwise only g vars
                # ??
                # self.train_template()
                self.train()
                print('Finish Training')

if __name__ == '__main__':
    cfg = FTConfig()
    model = FT(cfg)
    # model.debug_ds()
    model.run()

