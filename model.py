# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description: LSTM+CRF
"""
import tensorflow as tf


class BiLSTM_CRF(object):

    def __init__(self,config):
        self.config = config

    def build(self):
        self.add_placeholders()
        self.add_data_set()
        self.embedding_layer()
        self.biLSTM_layer()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.int32,[None,None],name='inputs')
        self.labels = tf.placeholder(tf.int32,[None,None],name='labels')
        self.lengths = tf.placeholder(tf.int32,[None],name='lengths')

        self.global_step = tf.Variable(0,dtype=tf.int32,trainable= False,name='global_step')

    def add_data_set(self):
        self.data_set = tf.data.Dataset\
            .from_tensor_slices((self.inputs,self.labels,self.lengths))\
            .shuffle(self.config.batch_size * 10)\
            .batch(self.config.batch_size)
        self.iter = self.data_set.make_initializable_iterator()
        self.input_batch,self.label_batch,self.length_batch = self.iter.get_next()

    def embedding_layer(self):
        with tf.device('/cpu:0'):
            embedding = tf.Variable(self.config.embedding_matrix,
                                    dtype=tf.float32,
                                    trainable=self.config.update_embedding,
                                    name='embedding')
            embeddding_inputs = tf.nn.embedding_lookup(embedding,
                                                       self.input_batch,
                                                       name='embedding_layer')
        self.embeddding_inputs = tf.nn.dropout(embeddding_inputs,self.config.dropout)

    def biLSTM_layer(self):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim)
        (output_fw_seq,output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                          cell_bw,
                                                                          inputs=self.embeddding_inputs,
                                                                          sequence_length=self.length_batch,
                                                                          dtype=tf.float32)
        output = tf.concat([output_fw_seq,output_bw_seq],axis=-1)
        output = tf.nn.dropout(output,self.config.dropout)

        W = tf.get_variable(name='W',
                            shape=[2 * self.config.hidden_dim,self.config.num_tags],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name='b',
                            shape=[self.config.num_tags],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        s = tf.shape(output)
        output = tf.reshape(output,[-1,2*self.config.hidden_dim])
        pred = tf.matmul(output,W) + b
        self.logits = tf.reshape(pred,[-1,s[1],self.config.num_tags])

    def loss_op(self):
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                   tag_indices=self.label_batch,
                                                                                   sequence_lengths=self.length_batch)
        self.loss = -tf.reduce_mean(log_likelihood,name='loss')

    def trainstep_op(self):
        if self.config.optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        self.train_op = optim.minimize(self.loss,global_step=self.global_step)

    def init_op(self):
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        self.init_var = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep= 4)

    def train(self,train_data,val_data):
        with tf.Session(config=self.sess_config) as sess:
            sess.run(self.init_var)

            for epoch in range(1,self.config.epoch):
                sess.run(self.iter.initializer, feed_dict={self.inputs: train_data.sentences,
                                                           self.labels: train_data.labels,
                                                           self.lengths: train_data.lengths})
                while True:
                    try:
                        sess.run(self.train_op)
                        loss = sess.run(self.loss)
                        print('epoch{}:loss is {}'.format(epoch, loss))
                        self.saver.save(sess,self.config.out_path,global_step=self.config.global_step)
                    except tf.errors.OutOfRangeError:
                        break


    def predict(self,sess,data):
        labels = []
        seqs = []
        sess.run(self.iter.initializer, feed_dict={self.inputs: data.sentences,
                                                   self.labels: data.labels,
                                                   self.lengths: data.lengths})
        while True:
            try:
                seq_len_list,logits,transition_params = sess.run([self.length_batch,self.logits,self.transition_params])
                label_list = []
                for logit,seq_len in zip(logits,seq_len_list):
                    viterbi_seq,_ = tf.contrib.crf.viterbi_decode(logit[:seq_len],transition_params)
                    label_list.append(viterbi_seq)
                labels.extend(label_list)
                seqs.extend(seq_len_list)
            except tf.errors.OutOfRangeError:
                break

        return labels,seqs


