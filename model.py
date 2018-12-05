# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description: LSTM+CRF
"""
import tensorflow as tf
import numpy as np
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
        self.crf_pred()
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
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell = cell_fw,output_keep_prob=self.config.dropout)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell = cell_bw,output_keep_prob=self.config.dropout)

        (output_fw_seq,output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                          cell_bw,
                                                                          inputs=self.embeddding_inputs,
                                                                          sequence_length=self.length_batch,
                                                                          dtype=tf.float32)
        output = tf.concat([output_fw_seq,output_bw_seq],axis=-1)

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

    def crf_pred(self):
        self.decode_tags, _ = tf.contrib.crf.crf_decode(self.logits, self.transition_params, self.length_batch)

    def trainstep_op(self):
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.config.lr,
                                                   global_step=self.global_step,
                                                   decay_rate=self.config.decay_rate,
                                                   decay_steps=self.config.decay_steps,
                                                   staircase= True)
        if self.config.optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

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
                        loss,global_step,lr = sess.run([self.loss,self.global_step,self.learning_rate])
                        print('epoch=>{},step=>{},lr=>{}:loss is {}'.format(epoch,global_step,lr,loss))
                        self.saver.save(sess,os.path.join(self.config.out_path,self.config.model_name),global_step=self.config.global_step)
                    except tf.errors.OutOfRangeError:
                        break

                predictions, prediction_lens = self.predict(sess, val_data.sentences,val_data.lengths)
                p, r, f ,_= self.evaluate(predictions, prediction_lens, val_data.labels)
                print('epoch{}:验证集\nP:{}  R:{}  F:{}'.format(epoch,p, r, f))
                self.save_metric(epoch,p,r,f)


    def test(self, test_data):
        with tf.Session(config=self.sess_config) as sess:
            print('==== testing ====')
            self.saver.restore(sess,tf.train.latest_checkpoint(self.config.out_path))
            predictions, prediction_lens = self.predict(sess, test_data.sentences, test_data.lengths)
            p, r, f, error_list = self.evaluate(predictions,prediction_lens,test_data.labels)
            print('P:{}\nR:{}\nF:{}'.format(p,r,f))
        return error_list

    def demo(self, demo_data):
        with tf.Session(config=self.sess_config) as sess:
            print('==== demo ====')
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.out_path))
            predictions, _ = self.predict(sess, demo_data.sentences, demo_data.lengths)
        return predictions

    def predict(self,sess,sentences,lengths):
        predictions =[]
        batch_size = self.config.batch_size
        index_l = 0
        index_r = batch_size
        while index_l < sentences.shape[0]:
            prediction = sess.run(self.decode_tags, feed_dict={self.input_batch: sentences[index_l:index_r],
                                                               self.length_batch: lengths[index_l:index_r]})
            predictions.extend(prediction)
            index_l += batch_size
            index_r = min(index_r+batch_size,sentences.shape[0])

        # sess.run(self.iter.initializer, feed_dict={self.inputs: data.sentences,
        #                                            self.labels: data.labels,
        #                                            self.lengths: data.lengths})
        # while True:
        #     try:
        #         seq_len_list,labels,logits,transition_params = sess.run([self.length_batch,self.label_batch,self.logits,self.transition_params])
        #         label_list = []
        #         for logit,seq_len in zip(logits,seq_len_list):
        #             viterbi_seq,_ = tf.contrib.crf.viterbi_decode(logit[:seq_len],transition_params)
        #             label_list.append(viterbi_seq)
        #         predictions.extend(label_list)
        #         prediction_lens.extend(seq_len_list)
        #         golds.extend(labels)
        #     except tf.errors.OutOfRangeError:
        #         break

        return predictions,lengths

    def evaluate(self,predictions,prediction_lens,golds):
        right = 0
        prediction_num = 0
        gold_num = 0
        error_list = []
        for prediction,gold,lens,index in zip(predictions,golds,prediction_lens,range(99999)):
            prediction_index = self.split_entity(prediction,lens)
            gold_index = self.split_entity(gold,lens)
            prediction_num += len(prediction_index)
            gold_num += len(gold_index)
            right += len([l for l in prediction_index if l in gold_index])
            if (len(prediction_index) != len(gold_index)) or(len([l for l in prediction_index if l in gold_index]) != len(prediction_index)):
                error = {
                    'index' : index,
                    'prediction_index' : prediction_index,
                    'gold_index' : gold_index
                }
                error_list.append(error)
        if gold_num == 0 or prediction_num == 0 or right == 0:
            return 0, 0, 0,error_list
        p = right/prediction_num
        r = right/gold_num
        f = 2*(p * r)/(p + r)
        return  p,r,f,error_list

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

    def save_metric(self,epoch,p,r,f):
        path = os.path.join(self.config.out_path,self.config.metric_name)
        with open(path,'a',encoding='utf-8') as fw:
            fw.write('{}\t{}\t{}\t{}\n'.format(epoch,p,r,f))

    def export(self,path):
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        with tf.Session(config=self.sess_config) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.out_path))

            input_info = tf.saved_model.utils.build_tensor_info(self.input_batch)
            length_info = tf.saved_model.utils.build_tensor_info(self.length_batch)
            output_info = tf.saved_model.utils.build_tensor_info(self.decode_tags)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': input_info,
                            'length': length_info},
                    outputs={'pred': output_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={
                                                     'predict': prediction_signature
                                                 })

            builder.save()





