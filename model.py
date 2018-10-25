# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description: LSTM+CRF
"""
import tensorflow as tf
import os


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
        if self.config.isdemo :
            self.data_set = tf.data.Dataset\
                .from_tensor_slices((self.inputs,self.labels,self.lengths))\
                .batch(self.config.batch_size)\
                .repeat(1)
        else:
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
            if self.config.retrain:
                sess.run(self.init_var)
            else:
                self.saver.restore(sess, tf.train.latest_checkpoint(self.config.out_path))

            for epoch in range(1,self.config.epoch):
                sess.run(self.iter.initializer, feed_dict={self.inputs: train_data.sentences,
                                                           self.labels: train_data.labels,
                                                           self.lengths: train_data.lengths})
                while True:
                    try:
                        sess.run(self.train_op)
                        loss,global_step = sess.run([self.loss,self.global_step])
                        print('epoch=>{},step=>{}:loss is {}'.format(epoch,global_step,loss))
                        self.saver.save(sess,os.path.join(self.config.out_path,self.config.model_name),global_step=self.config.global_step)
                    except tf.errors.OutOfRangeError:
                        break

                predictions, prediction_lens, golds = self.predict(sess, val_data)
                p, r, f = self.evaluate(predictions, prediction_lens, golds)
                print('epoch{}:验证集\nP:{}  R:{}  F:{}'.format(epoch,p, r, f))


    def test(self, test_data):
        with tf.Session(config=self.sess_config) as sess:
            print('==== testing ====')
            self.saver.restore(sess,tf.train.latest_checkpoint(self.config.out_path))
            predictions, prediction_lens, golds = self.predict(sess, test_data)
            p, r, f = self.evaluate(predictions,prediction_lens,golds)
            print('P:{}\nR:{}\nF:{}'.format(p,r,f))

    def demo(self, demo_data):
        with tf.Session(config=self.sess_config) as sess:
            print('==== demo ====')
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.out_path))
            predictions, _, _ = self.predict(sess, demo_data)
        return predictions

    def predict(self,sess,data):
        predictions = []
        prediction_lens = []
        golds = []
        sess.run(self.iter.initializer, feed_dict={self.inputs: data.sentences,
                                                   self.labels: data.labels,
                                                   self.lengths: data.lengths})
        while True:
            try:
                seq_len_list,labels,logits,transition_params = sess.run([self.length_batch,self.label_batch,self.logits,self.transition_params])
                label_list = []
                for logit,seq_len in zip(logits,seq_len_list):
                    viterbi_seq,_ = tf.contrib.crf.viterbi_decode(logit[:seq_len],transition_params)
                    label_list.append(viterbi_seq)
                predictions.extend(label_list)
                prediction_lens.extend(seq_len_list)
                golds.extend(labels)
            except tf.errors.OutOfRangeError:
                break

        return predictions,prediction_lens,golds

    def evaluate(self,predictions,prediction_lens,golds):
        right = 0
        prediction_num = 0
        gold_num = 0
        for prediction,gold,lens in zip(predictions,golds,prediction_lens):
            prediction_index = self.split_entity(prediction,lens)
            gold_index = self.split_entity(gold,lens)
            prediction_num += len(prediction_index)
            gold_num += len(gold_index)
            right += len([l for l in prediction_index if l in gold_index])
        if gold_num == 0 or prediction_num == 0 or right == 0:
            return 0, 0, 0
        p = right/prediction_num
        r = right/gold_num
        f = 2*(p * r)/(p + r)
        return  p,r,f

    def split_entity(self,sentence, lens):
        flag = False
        start = 0
        end = 0
        index = []
        for i in range(lens):
            if sentence[i] % 2 == 1 and flag == False:
                flag = True
                start = i
            elif sentence[i] % 2 == 1 and flag == True:
                flag = True
                end = i - 1
                index.append((start, end))
                start = i
            elif sentence[i] == 0 and flag == True:
                flag = False
                end = i - 1
                index.append((start, end))
            elif i == (lens - 1) and flag == True:
                flag = False
                end = i
                index.append((start, end))
        return index








