import tensorflow as tf
from tensorflow.python.ops import math_ops
from config import USE_DATA

SEED=1234
tf.set_random_seed(SEED)
USER_EMBEDDING_SIZE=128
ITEM_EMBEDDING_SIZE=128
OTHER_EMBEDDING_SIZE=32
LSTM_1=128
FC1_SIZE=128
FC2_SIZE=64
FC3_SIZE=1
MAX_TO_KEEP=10
#ml-1m
USER_SIZE=6040
ITEM_SIZE=3706
GENDER_DIM=2
AGE_DIM=7
OCCUPATION_DIM=21
RATE_DIM=6
YEAR_DIM=81
GENRE_DIM=25
DIRECTOR_DIM=2186
TOT_ITEM_EMBEDDING_SIZE=ITEM_EMBEDDING_SIZE+OTHER_EMBEDDING_SIZE*4
if USE_DATA=='bookcrossing':
    USER_SIZE=7369
    ITEM_SIZE = 291537
    AGE_DIM = 7
    AUTHOR_DIM=102031
    YEAR_DIM=118
    PUBLISHER_DIM=16810
    TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 3
elif USE_DATA=='avazu':
    USER_SIZE=34452
    ITEM_SIZE = 1294660
    DEVICE_TYPE_SIZE = 4
    C1_DIM=7
    C14_DIM=2626
    C15_DIM=8
    C16_DIM = 9
    TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 4

class ourModel(object):
    def __init__(self,mylog,cur_status,args):
        self.log=mylog
        self.cur_status=cur_status
        self.learning_rate=args.learning_rate
        self.user = tf.placeholder(tf.int32, [None, None])  # size:[batch_size,user_features]
        self.target_item = tf.placeholder(tf.int32, [None, None])  # size:[batch_size,features]
        self.target_label = tf.placeholder(tf.float32, [None, ])  # size:[batch_size]
        self.hist_items = tf.placeholder(tf.int32, [None, None, None])  # size:[batch_size,N,features]
        self.hist_items_y=tf.placeholder(tf.int32, [None, None])# size:[batch_size,N]
        self.items_length = tf.placeholder(tf.int32, [None, ])  # size:[batch_size]
        #self.learning_rate=tf.placeholder(tf.float32)
        self.batches=args.batches
        self.decay_rate=args.decay_rate
        self.trainable=args.trainable
        self.drop_keep_prob=args.drop_keep_prob
        self.regularizer_weight_decay=args.regularizer_weight_decay

        self.loss_weight_alpha=args.loss_weight_alpha
        global USER_EMBEDDING_SIZE,ITEM_EMBEDDING_SIZE,OTHER_EMBEDDING_SIZE,LSTM_1,TOT_ITEM_EMBEDDING_SIZE
        USER_EMBEDDING_SIZE=args.USER_EMBEDDING_SIZE
        ITEM_EMBEDDING_SIZE=args.ITEM_EMBEDDING_SIZE
        OTHER_EMBEDDING_SIZE=args.OTHER_EMBEDDING_SIZE
        LSTM_1=args.LSTM_1
        TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 4
        if USE_DATA == 'bookcrossing':
            TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 3
        elif USE_DATA == 'avazu':
            TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 4


        if args.regularizer=='l2':
            self.regularizer=tf.contrib.layers.l2_regularizer(self.regularizer_weight_decay)
        elif args.regularizer=='l1':
            self.regularizer = tf.contrib.layers.l1_regularizer(self.regularizer_weight_decay)
        else:
            self.regularizer=None
        self.activation=self.Dice
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)

    def build_model(self):
        with tf.variable_scope("embedding"):
            if USE_DATA == 'ml-1m':
                user_emb_w = tf.get_variable("user_emb_w", [USER_SIZE + 1, USER_EMBEDDING_SIZE],
                                             trainable=self.trainable)
                item_emb_w = tf.get_variable("item_emb_w", [ITEM_SIZE + 1, ITEM_EMBEDDING_SIZE],
                                             trainable=self.trainable)
                gender_emb_w = tf.get_variable("gender_emb_w", [GENDER_DIM + 1, OTHER_EMBEDDING_SIZE],
                                               trainable=self.trainable)
                age_emb_w = tf.get_variable("age_emb_w", [AGE_DIM + 1, OTHER_EMBEDDING_SIZE],
                                            trainable=self.trainable)
                occupation_emb_w = tf.get_variable("occupation_emb_w", [OCCUPATION_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                   trainable=self.trainable)
                item_rate_emb_w = tf.get_variable("rate_emb_w", [RATE_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                  trainable=self.trainable)
                item_year_emb_w = tf.get_variable("year_emb_w", [YEAR_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                  trainable=self.trainable)
                item_genre_emb_w = tf.get_variable("genre_emb_w", [GENRE_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                   trainable=self.trainable)
                item_director_emb_w = tf.get_variable("director_emb_w", [DIRECTOR_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                      trainable=self.trainable)

                user_embedding = tf.concat(values=[tf.nn.embedding_lookup(user_emb_w, self.user[:, 0]),
                                                   tf.nn.embedding_lookup(gender_emb_w, self.user[:, 1]),
                                                   tf.nn.embedding_lookup(age_emb_w, self.user[:, 2]),
                                                   tf.nn.embedding_lookup(occupation_emb_w, self.user[:, 3])],
                                           axis=1)
                self.log.info('user_embedding_size:{}'.format(user_embedding.get_shape().as_list()))

                target_embedding = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.target_item[:, 0]),
                                                     tf.nn.embedding_lookup(item_rate_emb_w, self.target_item[:, 1]),
                                                     tf.nn.embedding_lookup(item_year_emb_w, self.target_item[:, 2]),
                                                     tf.nn.embedding_lookup(item_genre_emb_w, self.target_item[:, 3]),
                                                     tf.nn.embedding_lookup(item_director_emb_w,
                                                                            self.target_item[:, 4])],
                                             axis=1)
                self.log.info('target_embedding_size:{}'.format(target_embedding.get_shape().as_list()))
                # size[batch_size,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]

                hist_embedding = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.hist_items[:, :, 0]),
                                                   tf.nn.embedding_lookup(item_rate_emb_w, self.hist_items[:, :, 1]),
                                                   tf.nn.embedding_lookup(item_year_emb_w, self.hist_items[:, :, 2]),
                                                   tf.nn.embedding_lookup(item_genre_emb_w, self.hist_items[:, :, 3]),
                                                   tf.nn.embedding_lookup(item_director_emb_w,
                                                                          self.hist_items[:, :, 4])], axis=2)
                self.log.info('hist_embedding_size1:{}'.format(hist_embedding.get_shape().as_list()))
            elif USE_DATA == 'bookcrossing':
                user_emb_w = tf.get_variable("user_emb_w", [USER_SIZE + 1, USER_EMBEDDING_SIZE],
                                             trainable=self.trainable)
                user_age_emb_w = tf.get_variable("age_emb_w", [AGE_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                 trainable=self.trainable)
                item_emb_w = tf.get_variable("item_emb_w", [ITEM_SIZE + 1, ITEM_EMBEDDING_SIZE],
                                             trainable=self.trainable)
                item_author_emb_w = tf.get_variable("author_emb_w", [AUTHOR_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                    trainable=self.trainable)
                item_year_emb_w = tf.get_variable("year_emb_w", [YEAR_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                  trainable=self.trainable)
                item_publisher_emb_w = tf.get_variable("publisher_emb_w", [PUBLISHER_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                       trainable=self.trainable)

                user_embedding = tf.concat(values=[tf.nn.embedding_lookup(user_emb_w, self.user[:, 0]),
                                                   tf.nn.embedding_lookup(user_age_emb_w, self.user[:, 1])],
                                           axis=1)
                self.log.info('user_embedding_size:{}'.format(user_embedding.get_shape().as_list()))

                target_embedding = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.target_item[:, 0]),
                                                     tf.nn.embedding_lookup(item_author_emb_w, self.target_item[:, 1]),
                                                     tf.nn.embedding_lookup(item_year_emb_w, self.target_item[:, 2]),
                                                     tf.nn.embedding_lookup(item_publisher_emb_w,
                                                                            self.target_item[:, 3])],
                                             axis=1)
                self.log.info('target_embedding_size:{}'.format(target_embedding.get_shape().as_list()))
                # size[batch_size,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]

                hist_embedding = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.hist_items[:, :, 0]),
                                                   tf.nn.embedding_lookup(item_author_emb_w, self.hist_items[:, :, 1]),
                                                   tf.nn.embedding_lookup(item_year_emb_w, self.hist_items[:, :, 2]),
                                                   tf.nn.embedding_lookup(item_publisher_emb_w,
                                                                          self.hist_items[:, :, 3])],
                                           axis=2)
                self.log.info('hist_embedding_size1:{}'.format(hist_embedding.get_shape().as_list()))
            elif USE_DATA == 'avazu':
                user_emb_w = tf.get_variable("user_emb_w", [USER_SIZE + 1, USER_EMBEDDING_SIZE],
                                             trainable=self.trainable)
                user_device_emb_w = tf.get_variable("device_emb_w", [DEVICE_TYPE_SIZE + 1, OTHER_EMBEDDING_SIZE],
                                                    trainable=self.trainable)
                item_emb_w = tf.get_variable("item_emb_w", [ITEM_SIZE + 1, ITEM_EMBEDDING_SIZE],
                                             trainable=self.trainable)
                item_c1_emb_w = tf.get_variable("c1_emb_w", [C1_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                trainable=self.trainable)
                item_c14_emb_w = tf.get_variable("c14_emb_w", [C14_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                 trainable=self.trainable)
                item_c15_emb_w = tf.get_variable("c15_emb_w", [C15_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                 trainable=self.trainable)
                item_c16_emb_w = tf.get_variable("c16_emb_w", [C16_DIM + 1, OTHER_EMBEDDING_SIZE],
                                                 trainable=self.trainable)

                user_embedding = tf.concat(values=[tf.nn.embedding_lookup(user_emb_w, self.user[:, 0]),
                                                   tf.nn.embedding_lookup(user_device_emb_w, self.user[:, 1])],
                                           axis=1)
                self.log.info('user_embedding_size:{}'.format(user_embedding.get_shape().as_list()))

                target_embedding = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.target_item[:, 0]),
                                                     tf.nn.embedding_lookup(item_c1_emb_w, self.target_item[:, 1]),
                                                     tf.nn.embedding_lookup(item_c14_emb_w, self.target_item[:, 2]),
                                                     tf.nn.embedding_lookup(item_c15_emb_w, self.target_item[:, 3]),
                                                     tf.nn.embedding_lookup(item_c16_emb_w, self.target_item[:, 4])],
                                             axis=1)
                self.log.info('target_embedding_size:{}'.format(target_embedding.get_shape().as_list()))
                # size[batch_size,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]

                hist_embedding = tf.concat(values=[tf.nn.embedding_lookup(item_emb_w, self.hist_items[:, :, 0]),
                                                   tf.nn.embedding_lookup(item_c1_emb_w, self.hist_items[:, :, 1]),
                                                   tf.nn.embedding_lookup(item_c14_emb_w, self.hist_items[:, :, 2]),
                                                   tf.nn.embedding_lookup(item_c15_emb_w, self.hist_items[:, :, 3]),
                                                   tf.nn.embedding_lookup(item_c16_emb_w, self.hist_items[:, :, 4])],
                                           axis=2)
                self.log.info('hist_embedding_size1:{}'.format(hist_embedding.get_shape().as_list()))

        with tf.variable_scope("attention"):
            # attention+sum_pooling
            # -- attention begin -------
            att_hist_embedding = self.attention(target_embedding, hist_embedding, self.items_length)
            self.log.info('hist_embedding_size2(after attention):{}'.format(hist_embedding.get_shape().as_list()))
            # -- attention end ---------
            att_hist_embedding = tf.layers.batch_normalization(inputs=att_hist_embedding,name='bn_hist',reuse=tf.AUTO_REUSE)
            att_hist_embedding = tf.reshape(att_hist_embedding, [-1, TOT_ITEM_EMBEDDING_SIZE])
            att_hist_embedding = tf.layers.dense(inputs=att_hist_embedding, units=TOT_ITEM_EMBEDDING_SIZE,name='fc_hist',
                                            activation = self.activation,kernel_regularizer = self.regularizer, trainable = self.trainable,reuse=tf.AUTO_REUSE)
            self.log.info('att_hist_embedding_size3:{}'.format(att_hist_embedding.get_shape().as_list()))
            # size[batch_size,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]


            #size=[batch_size,USER_EMBEDDING_SIZE*3]
        with tf.variable_scope("rnn"):
            rnn_hist_embedding = tf.layers.batch_normalization(inputs=hist_embedding, name='bn_hist', reuse=tf.AUTO_REUSE)
            cell_1 = tf.nn.rnn_cell.LSTMCell(LSTM_1)
            lstm_1, _ = tf.nn.dynamic_rnn(cell_1, rnn_hist_embedding, sequence_length=self.items_length, dtype=tf.float32,time_major=False)
            self.log.info('lstm_out_size:{}'.format(lstm_1.get_shape().as_list()))
            mask = tf.sequence_mask(self.items_length, tf.shape(lstm_1)[1], dtype=tf.float32)  # [batch_size,N]
            self.aux_loss=self.auxiliary_loss(lstm_1[:,:-1,:],hist_embedding[:,1:,:],mask[:,1:])
            mask = tf.expand_dims(mask, -1)  # [batch_size,N, 1]
            mask = tf.tile(mask,
                        [1, 1, tf.shape(lstm_1)[2]])  # [batch_size,N,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]
            lstm_1 *= mask  # [batch_size,N,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]
            lstm_1 = tf.reduce_sum(lstm_1, 1)
            lstm_1 = tf.div(lstm_1, tf.cast(tf.tile(tf.expand_dims(self.items_length, 1),
                                                [1, LSTM_1]), tf.float32))

        with tf.variable_scope("concat_mlp"):
            # concat
            tot_embedding = tf.concat(values=[user_embedding, att_hist_embedding, lstm_1, target_embedding], axis=1)
            self.log.info('tot_embedding_size:{}'.format(tot_embedding.get_shape().as_list()))
            #MLP
            tot_embedding=tf.layers.batch_normalization(inputs=tot_embedding, name='BN_MLP_input', reuse=tf.AUTO_REUSE)
            fc1_output=tf.layers.dense(inputs=tot_embedding,units=FC1_SIZE,name='FC_1',activation=self.activation,
                                       kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
            fc1_output = tf.nn.dropout(fc1_output, self.drop_keep_prob)
            self.log.info('fc1_output_size:{}'.format(fc1_output.get_shape().as_list()))
            fc2_output=tf.layers.dense(inputs=fc1_output,units=FC2_SIZE,name='FC_2',activation=self.activation,
                                       kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
            fc2_output = tf.nn.dropout(fc2_output, self.drop_keep_prob)
            self.log.info('fc2_output_size:{}'.format(fc2_output.get_shape().as_list()))
            fc3_output=tf.layers.dense(inputs=fc2_output,units=FC3_SIZE,name='FC_3',activation=None,
                                       kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
            self.log.info('fc3_output_size:{}'.format(fc3_output.get_shape().as_list()))
            fc3_output=tf.reshape(fc3_output,[-1])
            self.log.info('fc3_output_size_reshape:{}'.format(fc3_output.get_shape().as_list()))

            item_bias = tf.get_variable("item_bias", [ITEM_SIZE + 1], trainable=self.trainable,
                                        initializer=tf.constant_initializer(0.0))
            batch_bias = tf.gather(item_bias, self.target_item[:, 0])
            self.logits = batch_bias + fc3_output
            self.log.info('logits_size:{}'.format(self.logits.get_shape().as_list()))
            self.predicted = tf.sigmoid(self.logits)

        #loss
        self.global_step = tf.Variable(0, dtype=tf.int32,trainable=False, name='global_step')
        self.loss = tf.losses.log_loss(self.target_label, self.predicted)
        self.sum_loss=self.loss_weight_alpha*(self.loss+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))+(1-self.loss_weight_alpha)*self.aux_loss
        self.learning_rate=self.learning_rate
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step,self.batches*2,self.decay_rate)
        self.lr=tf.maximum(self.lr,tf.constant(1e-5,dtype=self.lr.dtype))
        
        trainable_params = tf.trainable_variables()
        self.local_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.global_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.sum_loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        loss_vars = tf.trainable_variables(scope='attention|rnn')
        local_gradients = tf.gradients(self.sum_loss, loss_vars)
        clip_local_gradients, _ = tf.clip_by_global_norm(local_gradients, 5)
        self.local_train_op = self.local_opt.apply_gradients(zip(clip_local_gradients, loss_vars),global_step=self.global_step)
        self.global_train_op = self.global_opt.apply_gradients(zip(clip_gradients, trainable_params),global_step=self.global_step)


    def auxiliary_loss(self,rnn_out,rnn_input,mask):
        mask = tf.cast(mask, tf.float32)
        aux_input = tf.concat([rnn_out, rnn_input], -1)
        y_prob=self.auxiliary_net(aux_input)
        self.log.info('aux_net_y_prob:{},mask:{},hist_y:{}'.format(y_prob.get_shape().as_list(),
                                                                   mask.get_shape().as_list(),
                                                                   self.hist_items_y.get_shape().as_list()))
        y_prob=tf.squeeze(y_prob)
        aux_loss=tf.losses.log_loss(self.hist_items_y[:,1:],tf.multiply(y_prob, mask))
        return aux_loss

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.sigmoid(dnn3)
        return y_hat

    def Dice(self,X, axis=-1, epsilon=0.000000001, name='dice'):
        with tf.variable_scope(name_or_scope='dice_activition', reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha' + name, X.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            beta = tf.get_variable('beta' + name, X.get_shape()[-1],
                                   initializer=tf.constant_initializer(0.0),
                                   dtype=tf.float32)
        input_shape = list(X.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

        # case: train mode (uses stats of the current batch)
        mean = tf.reduce_mean(X, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(X - brodcast_mean) + epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)
        x_normed = tf.layers.batch_normalization(X, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
        # x_normed = (X - brodcast_mean) / (brodcast_std + epsilon)
        x_p = tf.sigmoid(beta * x_normed)
        return alphas * (1.0 - x_p) * X + x_p * X

    def fc_layer(self,layer_name,input_tensor,W_shape,B_shape):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            fc_weights = tf.get_variable(layer_name+"_weights",W_shape,initializer=tf.truncated_normal_initializer())
            if self.regularizer!=None:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizer(fc_weights))
            fc_biases = tf.get_variable(layer_name+"_biases",B_shape, initializer=tf.constant_initializer())
            logits = tf.nn.sigmoid(tf.matmul(input_tensor, fc_weights) + fc_biases)
            if self.trainable:
                logits = tf.nn.dropout(logits, self.drop_keep_prob)
        return logits

    def attention(self,target_item,hist_items,items_length):
        hidden_units = target_item.get_shape().as_list()[-1]
        hist_items_length=tf.shape(hist_items)[1]
        target_item = tf.tile(target_item, [1,hist_items_length])
        target_item = tf.reshape(target_item, [-1,hist_items_length, hidden_units])
        din_all = tf.concat([target_item, hist_items, target_item - hist_items, target_item * hist_items], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all,FC1_SIZE, activation = self.activation, name='f1_att',
                                        kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, FC2_SIZE,activation = self.activation, name='f2_att',
                                        kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, FC3_SIZE, activation=None, name='f3_att',
                                        kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1,hist_items_length])
        outputs = d_layer_3_all
        # Mask
        key_masks = tf.sequence_mask(items_length, hist_items_length)  # [B, N]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, N]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, N]
        # Scale
        outputs = outputs / (hidden_units ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, N]
        # Weighted sum
        outputs = tf.matmul(outputs, hist_items,name="att_weighted_sum")  # [B, 1, H]
        self.log.info("attention output shape:{}".format(outputs.get_shape().as_list()))
        return outputs

    def attention_multi_targets_items(self,target_items, hist_items, items_length):
        '''
          target_items:     [B, NT, H] N is the number of ads
          hist_items:        [B, N, H]
          items_length: [B]
        '''
        hidden_units = target_items.get_shape().as_list()[-1]
        target_items_nums = target_items.get_shape().as_list()[1]
        hist_items_length = tf.shape(hist_items)[1]
        target_items = tf.tile(target_items, [1, 1,hist_items_length])
        target_items = tf.reshape(target_items,
                             [-1, target_items_nums,hist_items_length, hidden_units])  # shape : [B, N, T, H]
        hist_items = tf.tile(hist_items, [1, target_items_nums, 1])
        hist_items = tf.reshape(hist_items, [-1, target_items_nums, hist_items_length, hidden_units])  # shape : [B, N, T, H]
        din_all = tf.concat([target_items, hist_items, target_items - hist_items, target_items * hist_items], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, FC1_SIZE, activation = self.activation, name='f1_att',
                                        kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, FC2_SIZE, activation = self.activation, name='f2_att',
                                        kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, FC3_SIZE, activation=None, name='f3_att',
                                        kernel_regularizer=self.regularizer,trainable=self.trainable,reuse=tf.AUTO_REUSE)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, target_items_nums, 1, hist_items_length])
        outputs = d_layer_3_all
        # Mask
        key_masks = tf.sequence_mask(items_length, hist_items_length)  # [B, T]
        key_masks = tf.tile(key_masks, [1, target_items_nums])
        key_masks = tf.reshape(key_masks, [-1, target_items_nums, 1, hist_items_length])  # shape : [B, N, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]
        # Scale
        outputs = outputs / (hidden_units** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
        outputs = tf.reshape(outputs, [-1, 1, hist_items_length])
        keys = tf.reshape(hist_items, [-1, hist_items_length, hidden_units])
        # print outputs.get_shape().as_list()
        # print keys.get_sahpe().as_list()
        # Weighted sum
        outputs = tf.matmul(outputs, keys)
        outputs = tf.reshape(outputs, [-1, target_items_nums, hidden_units])  # [B, N, 1, H]
        self.log.info("muti attention output shape:{}".format(outputs.get_shape().as_list()))
        return outputs


def our_train_local_global(sess,model,cur_batch):
    users,targets,targets_labels,hists,items_length,hists_y=cur_batch[0],cur_batch[1],cur_batch[2],cur_batch[3],cur_batch[4],cur_batch[5]
    fetches = [model.sum_loss, model.global_step,model.lr, model.local_train_op]
    sum_loss=0.0
    if (len(users)//3)>=1:
        pos=-(len(users)//3)
    else:
        pos=-1
    feed_dict = {model.user: users[:pos], model.target_item: targets[:pos],
                 model.target_label: targets_labels[:pos], model.hist_items: hists[:pos],
                 model.items_length: items_length[:pos],model.hist_items_y:hists_y[:pos]}
    loss, step,lr, _ = sess.run(fetches, feed_dict)
    sum_loss += loss

    fetches = [model.sum_loss, model.global_step,model.lr, model.global_train_op]
    feed_dict = {model.user: users[pos:], model.target_item: targets[pos:],
                 model.target_label: targets_labels[pos:], model.hist_items: hists[pos:],
                 model.items_length: items_length[pos:],model.hist_items_y:hists_y[pos:]}
    loss, step,lr, _ = sess.run(fetches, feed_dict)
    sum_loss+=loss
    return sum_loss,step,lr
def our_train_global(sess,model,cur_batch):
    users,targets,targets_labels,hists,items_length,hists_y=cur_batch[0],cur_batch[1],cur_batch[2],cur_batch[3],cur_batch[4],cur_batch[5]

    fetches = [model.sum_loss, model.global_step,model.lr, model.global_train_op]
    feed_dict = {model.user: users, model.target_item: targets,
                 model.target_label: targets_labels, model.hist_items: hists,
                 model.items_length: items_length,model.hist_items_y:hists_y}
    loss, step,lr, _ = sess.run(fetches, feed_dict)
    return loss,step,lr
    
def our_eval_global(sess,model,cur_batch):
    users, targets, targets_labels, hists, items_length,hists_y = cur_batch[0], cur_batch[1], cur_batch[2], cur_batch[3], \
                                                          cur_batch[4],cur_batch[5]
    fetches = [model.target_label,model.predicted,model.sum_loss]
    feed_dict = {model.user: users, model.target_item: targets,
                 model.target_label: targets_labels, model.hist_items: hists,
                 model.items_length: items_length,model.hist_items_y:hists_y}
    cur_label, cur_predicted,loss= sess.run(fetches, feed_dict)
    return cur_label, cur_predicted, loss

def our_eval_global_finetune(sess,model,cur_batch):
    users, targets, targets_labels, hists, items_length,hists_y = cur_batch[0], cur_batch[1], cur_batch[2], cur_batch[3], \
                                                          cur_batch[4],cur_batch[5]
    fetches = [model.target_label,model.predicted,model.sum_loss,model.global_train_op]
    feed_dict = {model.user: users, model.target_item: targets,
                 model.target_label: targets_labels, model.hist_items: hists,
                 model.items_length: items_length,model.hist_items_y:hists_y}
    cur_label, cur_predicted,loss,_= sess.run(fetches, feed_dict)
    return cur_label, cur_predicted, loss
