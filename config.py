# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description:
"""

class model_config(object):
    def __init__(self):
        self.train_data = 'resource/data/corpus/train.txt'
        self.val_data = 'resource/data/corpus/val.txt'
        self.test_data = 'resource/data/corpus/test.txt'
        self.demo_data = 'resource/data/corpus/demo.txt'
        self.embedding_path = 'resource/data/word2id/embedding_matrix.cpkt'
        self.word2id_path = 'resource/data/word2id/word2id.cpkt'
        self.out_path = 'resource/result'
        self.model_name = 'test-model'
        self.export_path = 'resource/result/EMR_NER/0001'
        self.metric_name = 'metric.txt'
        self.error_text = 'error.txt'
        self.isdemo = True

        self.retrain = True
        self.epoch = 100
        self.batch_size = 64
        self.hidden_dim = 256
        self.optimizer = 'Adam'
        self.CRF = True
        self.lr = 0.001
        self.dropout = 0.2
        self.update_embedding = True
        self.num_tags = 9
        self.global_step=500

        self.embedding_matrix = None
