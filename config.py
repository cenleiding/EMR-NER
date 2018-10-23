# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description:
"""

class model_config(object):
    def __init__(self):
        self.train_data = 'resourse/data/corpus/train.txt'
        self.val_data = 'resourse/data/corpus/val.txt'
        self.test_data = 'resourse/data/corpus/test.txt'
        self.embedding_path = 'resourse/data/word2id/embedding_matrix.cpkt'
        self.word2id_path = 'resourse/data/word2id/word2id.cpkt'
        self.out_path = 'resourse/result/test-model'

        self.epoch = 40
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
