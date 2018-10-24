# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description:
"""
import pickle
import numpy as np

class DataSet(object):

    def __init__(self,corpus_path,word2id_path,embedding_path):
        self.tag2label = {'O': 0,
                          'B-NAME': 1, 'I-NAME': 2,
                          'B-ADDRESS': 3, 'I-ADDRESS': 4,
                          'B-ORGANIZATION': 5, 'I-ORGANIZATION': 6,
                          'B-DETAIL': 7, 'I-DETAIL': 8}
        self.word2id = read_word2id(word2id_path)
        self.embeddding = read_embeddding(embedding_path)
        (self.sentences_origin, tags) = read_corpus(corpus_path)
        (self.sentences, self.labels, self.lengths) = sentences_tags2id(self.sentences_origin,tags,self.word2id,self.tag2label)

def read_corpus(corpus_path):
    sentences,tags=[],[]
    with open(corpus_path,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_,tag_ = [],[]
    for line in lines:
        if line != '\n':
            [char,label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            sentences.append(sent_)
            tags.append(tag_)
            sent_,tag_ = [],[]
    return (sentences,tags)

def read_word2id(word2id_path):
    with open(word2id_path,'rb') as fr:
        word2id=pickle.load(fr)
    return word2id

def read_embeddding(embedding_path):
    embedding_matrix = np.load(embedding_path)
    return embedding_matrix

def sentences_tags2id(sentences,tags,word2id,tag2label,max_length=400):
    sentences_id = []
    tags_id = []
    lengths = []
    for sentence in sentences:
        sentence_id = []
        for word in sentence:
            if word.isdigit():
                word = 'NUM'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = 'ENG'
            if word not in word2id:
                word = 'UNK'
            sentence_id.append(word2id[word])
        sentences_id.append(sentence_id)
        lengths.append(len(sentence_id))

    for tag in tags:
        tag_id = []
        for t in tag:
            tag_id.append(tag2label.get(t))
        tags_id.append(tag_id)

    #统一句子长度,放弃过长的句子，并转为np数组
    length = min(max_length,max(lengths))

    sentences_id_np = np.zeros([len(sentences_id),length])
    tags_id_np = np.zeros([len(tags_id),length])

    for i ,ids in enumerate(sentences_id):
        if lengths[i] <= length:
            sentences_id_np[i,:lengths[i]] = np.array(ids)
    for i ,ids in enumerate(tags_id):
        if lengths[i] <= length:
            tags_id_np[i,:lengths[i]] = np.array(ids)
        else:
            lengths[i] = 0

    return (sentences_id_np ,tags_id_np,lengths)

