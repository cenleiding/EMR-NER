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
import os

parser = argparse.ArgumentParser(description='BiLSTM-CRF 配置')
parser.add_argument('--train_data',type=str,default='resource/data/corpus/train.txt',help='训练集数据路径')
parser.add_argument('--val_data',type=str,default='resource/data/corpus/val.txt',help='验证集数据路径')
parser.add_argument('--test_data',type=str,default='resource/data/corpus/test.txt',help='测试集路径')
parser.add_argument('--embedding_path',type=str,default='resource/data/word2id/embedding_matrix.cpkt',help='嵌入矩阵 路径')
parser.add_argument('--word2id_path',type=str,default='resource/data/word2id/word2id.cpkt',help='word2id 路径')
parser.add_argument('--out_path',type=str,default='resource/result',help='模型保存地址')

parser.add_argument('--epoch',type=int,default=40,help='训练周期')
parser.add_argument('--batch_size',type=int,default=64,help='批处理大小')
parser.add_argument('--hidden_dim',type=int,default=256,help='隐藏节点维度')
parser.add_argument('--optimizer',type=str,default='Adam',choices=['Adam','SGD'],help="优化方式选择=》['Adam','SGD']")
parser.add_argument('--CRF',type=bool,default=True,help='True: CRF; False: Softmax')
parser.add_argument('--lr',type=float,default=0.001,help='训练速度')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout 值')
parser.add_argument('--update_embedding',type=bool,default=True,help='是否一起训练embeddding')

parser.add_argument('--mode',type=str,default='train',choices=['train','test','demo','export'],help='train:训练模型；test：测试模型；demo：使用模型')

args = parser.parse_args()



config = config.model_config()
# for arg in vars(args):
#     setattr(config,arg,getattr(args,arg))


if args.mode == 'train':
    train_data = data.DataSet(config.word2id_path,config.embedding_path)
    val_data = data.DataSet(config.word2id_path,config.embedding_path)
    train_data.read_tag_file(config.train_data)
    val_data.read_tag_file(config.val_data)
    config.embedding_matrix = train_data.embeddding
    model = model.BiLSTM_CRF(config)
    model.build()
    model.train(train_data,val_data)

if args.mode =='test':
    test_data = data.DataSet(config.word2id_path,config.embedding_path)
    test_data.read_tag_file(config.test_data)
    config.embedding_matrix = test_data.embeddding
    model = model.BiLSTM_CRF(config)
    model.build()
    error_list = model.test(test_data)
    with open(os.path.join(config.out_path,config.error_text),'w',encoding='utf-8')as fw:
        for error in error_list:
            fw.write('原句：{}\n'.format(''.join(test_data.sentences_origin[error['index']])))
            fw.write('标注：')
            for gold_index in error['gold_index']:
                fw.write('{}\t'.format(''.join(test_data.sentences_origin[error['index']][gold_index[0]:gold_index[1]+1])))
            fw.write('\n')
            fw.write('预测：')
            for prediction_index in error['prediction_index']:
                fw.write('{}\t'.format(''.join(test_data.sentences_origin[error['index']][prediction_index[0]:prediction_index[1]+1])))
            fw.write('\n')
            fw.write('\n')

if args.mode =='demo':
    result = {
        'name' : [],
        'address' : [],
        'organization' : [],
        'detail' : []
    }
    demo_data = data.DataSet(config.word2id_path,config.embedding_path)
    demo_data.read_demo_file(config.demo_data)
    config.embedding_matrix = demo_data.embeddding
    model = model.BiLSTM_CRF(config)
    model.build()
    predictions = model.demo(demo_data)
    for prediction,sentence in zip(predictions,demo_data.sentences_origin):
        index = [i for i in range(len(prediction)) if prediction[i]%2 ==1]
        for i in index:
            start = i
            end = i
            while (end+1)<len(prediction) and prediction[end+1] == prediction[start]+1:
                end +=1
            if prediction[start] == 1:
                result['name'].append(sentence[start:end + 1])
            if prediction[start] == 3:
                result['address'].append(sentence[start:end + 1])
            if prediction[start] == 5:
                result['organization'].append(sentence[start:end + 1])
            if prediction[start] == 7:
                result['detail'].append(sentence[start:end + 1])
    print('原文：\n{}'.format(''.join(map(lambda sen:''.join(sen)+'\n',demo_data.sentences_origin))))
    print('处理结果：\n')
    print('name:{}'.format(set(map(lambda s:''.join(s),result['name']))))
    print('address:{}'.format(set(map(lambda s:''.join(s),result['address']))))
    print('organization:{}'.format(set(map(lambda s:''.join(s),result['organization']))))
    print('detail:{}'.format(set(map(lambda s:''.join(s),result['detail']))))

if args.mode =='export':
    embeddding = data.DataSet(config.word2id_path,config.embedding_path)
    config.embedding_matrix = embeddding.embeddding
    model = model.BiLSTM_CRF(config)
    model.build()
    model.export(config.export_path)




