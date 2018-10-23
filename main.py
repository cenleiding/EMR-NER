# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/10/22
@description:
"""
import data
import argparse
import config
import model

parser = argparse.ArgumentParser(description='BiLSTM-CRF 配置')
parser.add_argument('--train_data',type=str,default='resourse/data/corpus/train.txt',help='训练集数据路径')
parser.add_argument('--val_data',type=str,default='resourse/data/corpus/val.txt',help='验证集数据路径')
parser.add_argument('--test_data',type=str,default='resourse/data/corpus/test.txt',help='测试集路径')
parser.add_argument('--embedding_path',type=str,default='resourse/data/word2id/embedding_matrix.cpkt',help='嵌入矩阵 路径')
parser.add_argument('--word2id_path',type=str,default='resourse/data/word2id/word2id.cpkt',help='word2id 路径')
parser.add_argument('--out_path',type=str,default='resourse/result',help='模型保存地址')

parser.add_argument('--epoch',type=int,default=40,help='训练周期')
parser.add_argument('--batch_size',type=int,default=64,help='批处理大小')
parser.add_argument('--hidden_dim',type=int,default=256,help='隐藏节点维度')
parser.add_argument('--optimizer',type=str,default='Adam',choices=['Adam','SGD'],help="优化方式选择=》['Adam','SGD']")
parser.add_argument('--CRF',type=bool,default=True,help='True: CRF; False: Softmax')
parser.add_argument('--lr',type=float,default=0.001,help='训练速度')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout 值')
parser.add_argument('--update_embedding',type=bool,default=True,help='是否一起训练embeddding')

parser.add_argument('--mode',type=str,default='train',choices=['train','test','demo'],help='train:训练模型；test：测试模型；demo：使用模型')

args = parser.parse_args()



config = config.model_config()
# for arg in vars(args):
#     setattr(config,arg,getattr(args,arg))


if args.mode == 'train':
    train_data = data.DataSet(config.train_data,config.word2id_path,config.embedding_path)
    val_data = data.DataSet(config.val_data,config.word2id_path,config.embedding_path)
    config.embedding_matrix = train_data.embeddding
    model = model.BiLSTM_CRF(config)
    model.build()
    model.train(train_data,val_data)

