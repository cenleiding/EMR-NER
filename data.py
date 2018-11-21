# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description:
"""
import pickle
import numpy as np
import re

class DataSet(object):

    def __init__(self,word2id_path,embedding_path):
        self.tag2label = {'O': 0,
                          'B-NAME': 1, 'I-NAME': 2,
                          'B-ADDRESS': 3, 'I-ADDRESS': 4,
                          'B-ORGANIZATION': 5, 'I-ORGANIZATION': 6,
                          'B-DETAIL': 7, 'I-DETAIL': 8}
        self.word2id = read_word2id(word2id_path)
        self.embeddding = read_embeddding(embedding_path)

    def read_tag_file(self,corpus_path):
        (self.sentences_origin, tags) = read_corpus(corpus_path)
        (self.sentences, self.labels, self.lengths, self.sentences_origin) = sentences_tags2id(self.sentences_origin, tags, self.word2id,self.tag2label)

    def read_demo_file(self,demo_path):
        (self.sentences_origin,tags) = read_demo(demo_path)
        (self.sentences, self.labels, self.lengths, self.sentences_origin) = sentences_tags2id(self.sentences_origin, tags, self.word2id,self.tag2label)

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
        elif len(sent_)!=0:
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

def sentences_tags2id(sentences,tags,word2id,tag2label,max_length=250):
    sentences_id = []
    tags_id = []
    lengths_id = []
    lengths = []
    new_sentenses = []
    for sentence in sentences:
        sentence_id = []
        for word in sentence:
            if word not in word2id:
                word = 'UNK'
            sentence_id.append(word2id[word])
        sentences_id.append(sentence_id)
        lengths_id.append(len(sentence_id))

    for tag in tags:
        tag_id = []
        for t in tag:
            tag_id.append(tag2label.get(t))
        tags_id.append(tag_id)

    #统一句子长度,切割过长的句子，并转为np数组

    sentences_id_np = np.zeros([max_length])
    tags_id_np = np.zeros([max_length])
    for l ,s_ids,t_ids,s in zip(lengths_id,sentences_id,tags_id,sentences):
        new_l = l
        while new_l >max_length:
            new_sentense_id = np.array(s_ids[l - new_l : l - new_l + max_length])
            new_tag = np.array(t_ids[l - new_l : l - new_l + max_length])
            new_sentenses.append(s[l - new_l : l - new_l + max_length])
            sentences_id_np = np.row_stack((sentences_id_np,new_sentense_id))
            tags_id_np = np.row_stack((tags_id_np,new_tag))
            new_l = new_l -max_length
            lengths.append(max_length)
        new_sentense_id = np.zeros([max_length])
        new_sentense_id[:new_l]=np.array(s_ids[l - new_l:l])
        new_tag = np.zeros([max_length])
        new_tag[:new_l] = np.array(t_ids[l - new_l:l])
        new_sentenses.append(s[l - new_l:l])
        sentences_id_np = np.row_stack((sentences_id_np, new_sentense_id))
        tags_id_np = np.row_stack((tags_id_np, new_tag))
        lengths.append(new_l)
    sentences_id_np = sentences_id_np[1:,:]
    tags_id_np = tags_id_np[1:,:]
    return (sentences_id_np ,tags_id_np,lengths,new_sentenses)

def read_demo(demo_path):
    sentences =[]
    tags = []
    with open(demo_path,'r',encoding='utf-8') as fr:
        text = fr.read()
    #中文标点转英文标点
    chinese_punctuation = '，【】“”‘’！？（）１２３４５６７８９０'
    english_punctuation = ',[]""\'\'!?()1234567890'
    table = str.maketrans(chinese_punctuation, english_punctuation)
    text = text.translate(table)
    #全角转半角
    newtext = ''
    for s in text:
        num = ord(s)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xFEE0
        u = chr(num)
        newtext += u
    #去除空格\n\t
    newtext=re.sub(r'\s',"",newtext)
    sentence = []
    tag = []
    for char in newtext:
        if char=='。':
            sentences.append(sentence)
            sentence = []
            tags.append(tag)
            tag = []
        else:
            sentence.append(char)
            tag.append('O')
    return (sentences,tags)


