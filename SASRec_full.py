
# Default params
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default="Video")
# parser.add_argument('--train_dir', default="default")
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=201, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.5, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)

# Self-Attentive Sequential Recommendation
# https://github.com/kang205/SASRec
from Recommender import Recommender
from SASRecLayer import *
from keras_preprocessing.sequence import pad_sequences

class SASRec(Recommender):
    def __init__(self, usernum, itemnum, hidden_units=50, maxlen=50, num_blocks=2,
                 num_heads=1,
                 dropout_rate=0.5,
                 l2_emb=0.0, lr=0.001, reuse=None, args=None, eps=0.5, time_stamp=None):

        self.uNum = usernum + 1
        self.iNum = itemnum + 1
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.eps = args.eps
        self.reg_adv = args.reg_adv
        self.args = args
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # with tf.name_scope("SASRec"):
            # sequence embedding, item embedding table
            self.emb, self.item_emb_table = embedding(self.input_seq,
                                                      vocab_size=itemnum + 1,
                                                      num_units=hidden_units,
                                                      zero_pad=True,
                                                      scale=True,
                                                      l2_reg=l2_emb,
                                                      scope="input_embeddings",
                                                      with_t=True,
                                                      reuse=reuse
                                                      )

            self.delta_emb = tf.Variable(tf.zeros(shape=[itemnum + 1, hidden_units]), name='delta_emb',
                                         dtype=tf.float32, trainable=False)

            self.delta_E = tf.Variable(tf.zeros(shape=[1, self.args.batch_size, maxlen, hidden_units]),
                                       name='delta_emb', dtype=tf.float32, trainable=False)
            self.delta_T = tf.Variable(tf.zeros(shape=[1, self.args.batch_size, maxlen, hidden_units]),
                                       name='delta_emb', dtype=tf.float32, trainable=False)

            # self.delta_pos_emb = tf.Variable(tf.zeros(shape=[1, self.args.batch_size, maxlen, hidden_units]),
            #                              name='delta_pos_emb', dtype=tf.float32, trainable=False)

            # Positional Encoding
            self.t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=maxlen,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )


            self.seq = self.emb + self.t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(num_blocks):
                # with tf.variable_scope("num_blocks_%d" % i):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[hidden_units, hidden_units],
                                           dropout_rate=dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * maxlen])
        pos_emb = tf.nn.embedding_lookup(self.item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(self.item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(itemnum + 1))
        test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], maxlen, itemnum + 1])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        # if reuse is None:
        #     tf.summary.scalar('auc', self.auc)
        #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.98)
        #     self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # else:
        #     tf.summary.scalar('test_auc', self.auc)
        tf.summary.scalar('auc', self.auc)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.merged = tf.summary.merge_all()

        if args.adver:
            #
            # self.output_adv, embed_seq_pos = self._create_inference_adv(self.pos, maxlen, hidden_units)
            self.output_adv = self._create_inference_adv(pos, maxlen, hidden_units)
            # self.output_neg_adv, embed_seq_neg = self._create_inference_adv(self.neg, maxlen, hidden_units)
            self.output_neg_adv = self._create_inference_adv(neg, maxlen, hidden_units)
            # self.result_adv = tf.clip_by_value(self.output_adv - self.output_neg_adv, -80.0, 1e8)

            self.adv_loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.output_adv) + 1e-24) * istarget -
                tf.log(1 - tf.sigmoid(self.output_neg_adv) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
            reg_adv_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.adv_loss += sum(reg_adv_losses)

            self.opt_loss = self.loss + (self.reg_adv * self.adv_loss)
            self.train_op = self.optimizer.minimize(self.opt_loss, global_step=self.global_step)

            self._create_adversarial()

        # # initialized the save op
        #
        # if args.adver:
        #     self.ckpt_save_path = "Pretrain/%s/ASASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        #     self.ckpt_restore_path = "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        # else:
        #     self.ckpt_save_path = "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        #     self.ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, args.restore)
        #
        # if not os.path.exists(self.ckpt_save_path):
        #     os.makedirs(self.ckpt_save_path)
        # if self.ckpt_restore_path and not os.path.exists(self.ckpt_restore_path):
        #     os.makedirs(self.ckpt_restore_path)
        #
        # self.saver_ckpt = tf.train.Saver()
        #
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement = True
        # self.sess = tf.Session(config=config)
        #
        # self.sess.run(tf.initialize_all_variables())
        #
        # # restore the weights when pretrained
        # if args.restore is not None:
        #     ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_restore_path + 'checkpoint'))
        #     if ckpt and ckpt.model_checkpoint_path:
        #         self.saver_ckpt.restore(self.sess, ckpt.model_checkpoint_path)
        #
        # # initialize the weights
        # else:
        #     logging.info("Initialized from scratch")

    def _create_inference_adv(self, item_input, maxlen, hidden_units):
        emb = tf.nn.embedding_lookup(self.item_emb_table, item_input)
        # emb = tf.reduce_sum(tf.nn.embedding_lookup(self.item_emb_table, item_input), 1)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])

        emb_plus_delta = emb + tf.nn.embedding_lookup(self.delta_emb, item_input)

        return tf.reduce_sum(emb_plus_delta * seq_emb, -1)
        # return tf.reduce_sum(emb * seq_emb, -1)

    def _create_adversarial(self):
        self.grad_emb = tf.gradients(self.loss, [self.item_emb_table])
        self.grad_emb_dense = tf.stop_gradient(self.grad_emb[0])
        self.update_emb = self.delta_emb.assign(tf.nn.l2_normalize(self.grad_emb_dense, 1) * self.eps)


    def init(self, trainSeq, batch_size, sess):
        self.trainSeq = trainSeq
        self.sampler = WarpSampler(self.trainSeq, self.uNum, self.iNum, batch_size=batch_size, maxlen=self.maxlen,
                                   n_workers=3)
        self.sess = sess

        # self.saver_ckpt.save(self.sess, self.ckpt_save_path + 'weights', global_step=0)

    def rank(self, users, items):
        users = users[0]
        seq = pad_sequences([self.trainSeq[users[0]]], self.maxlen)

        score = self.sess.run(self.test_logits,
                              {self.u: users[0], self.input_seq: seq, self.test_item: range(self.iNum),
                               self.is_training: False})[0]
        # res = []
        # for i in items:
        #
        #     res.append(score)
        # return np.array(res)
        return score[items.flatten()]

    def save(self, path):
        pass

    def load_pre_train(self, pre):
        pass

    def get_train_instances(self, train):
        return None, None

    def train(self, x_train, y_train, batch_size):
        losses = []
        num_batch = int(len(self.trainSeq) / batch_size)

        for step in list(range(num_batch)):
            u, seq, pos, neg = self.sampler.next_batch()

            if self.args.adver:
                self.sess.run([self.update_emb], {self.u: u, self.input_seq: seq, self.pos: pos, self.neg: neg,
                                                  self.is_training: False})

                auc, loss, _ = self.sess.run([self.auc, self.opt_loss, self.train_op],
                                             {self.u: u, self.input_seq: seq, self.pos: pos, self.neg: neg,
                                              self.is_training: True})
            else:

                auc, loss, _ = self.sess.run([self.auc, self.loss, self.train_op],
                                             {self.u: u, self.input_seq: seq, self.pos: pos, self.neg: neg,
                                              self.is_training: True})

            losses.append(loss)
            # break

        return np.mean(losses)

    def get_params(self):
        return "_ml%d" % (self.maxlen)


