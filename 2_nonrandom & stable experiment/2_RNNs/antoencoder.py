import random

import pandas as pd
from keras.layers import Dense, LSTM, Input, concatenate, TimeDistributed, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
import keras
import cg_tm_kl
from data_helper import get_data, trans_sign
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD

from sklearn.metrics import classification_report, accuracy_score,recall_score,precision_score,f1_score
import numpy as np
import pandas
import os
import logger
from logger import Logger
def PathSlect(PATH):
    PathOutput = []
    for path in PATH:
        if path.find('StegaSeq')> 0:
            PathOutput.append(path)

    return PathOutput

# 建立基于LSTM自编码器模型
def build_autoEncoder(timeSteps, dimension):
    input = Input(shape=(timeSteps, dimension))
    # 编码器
    encoded1 = LSTM(32, activation='relu', return_sequences=True, name='encoded1', dropout=0.3)(input)
    encoded2 = LSTM(2, activation='relu', return_sequences=True, name='encoded2', dropout=0.3)(encoded1)

    # 解码器
    decoded1 = LSTM(2, activation='relu', return_sequences=True, name='decoded1', dropout=0.3)(encoded2)
    decoded2 = LSTM(32, activation='relu', return_sequences=True, name='decoded2', dropout=0.3)(decoded1)
    output = TimeDistributed(Dense(dimension))(decoded2)

    # 自编码器模型
    autoEncoder = Model(inputs=input, outputs=output)
    # encoder模型
    encoder = Model(inputs=input, outputs=encoded2)
    return autoEncoder, encoder


# 建立lSTM预测模型
def build_lstm(timeSteps, dimension):
    input = Input(shape=(timeSteps, dimension))
    lstm = LSTM(64, activation='relu', name='lstm', dropout=0.3)(input)
    dense1 = Dense(32, activation='relu')(lstm)
    dense = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=input, outputs=dense)
    return model


# 建立CNN预测模型
def build_cnn(timeSteps, dimension):
    input = Input(shape=(timeSteps, dimension))
    cnn1 = Conv1D(64, 4, padding='same', strides=1, activation='relu')(input)
    cnn1 = MaxPooling1D(pool_size=timeSteps - 4 + 1)(cnn1)
    cnn2 = Conv1D(64, 8, padding='same', strides=1, activation='relu')(input)
    cnn2 = MaxPooling1D(pool_size=timeSteps - 8 + 1)(cnn2)
    cnn3 = Conv1D(64, 12, padding='same', strides=1, activation='relu')(input)
    cnn3 = MaxPooling1D(pool_size=timeSteps - 12 + 1)(cnn3)

    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flatten = Flatten()(cnn)
    dropout1 = Dropout(0.3)(flatten)
    dense1 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.1)(dense1)
    dense2 = Dense(1, activation='sigmoid')(dropout2)
    model = Model(inputs=input, outputs=dense2)
    return model


def csv_shengc(pathsc,file_csv,ori, sample_num,Seqlength):
    dirs_csv = file_csv
    pathori = ori

    raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc= Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

    pathwrite_sc = r'D:\Desktop\seqs\Ho Bae\tt\1.txt'
    with open(pathwrite_sc, 'w') as f1:
        for line in raw_pos:
            f1.write(line)
            f1.write('\n')

    raw_neg = cg_tm_kl.txt_process_sc_duo(pathori, len_sc=Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

    pathwrite_ori = r'D:\Desktop\seqs\Ho Bae\tt\2.txt'
    with open(pathwrite_ori, 'w') as f2:
        for line in raw_neg:
            f2.write(line)
            f2.write('\n')

    pos = pd.read_csv(pathwrite_sc, header=None)
    neg = pd.read_csv(pathwrite_ori, header=None)

    x = np.concatenate((pos, neg))
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    save = pd.DataFrame(columns=['text', 'label'], index=range(len(x)))

    save['text'] = x
    save['label'] = y

    # t = pathsc[pathsc.find('traditionnal method') +len('traditionnal method') + 1  : len(pathsc)-4]

    t = pathsc[pathsc.rfind('\\'): len(pathsc) - 4]
    file_ = dirs_csv + t + '.csv'

    save.to_csv(file_)

def RNNs_Classifier(path):
    # 获取数据
    data_x_final, y = get_data(path)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(data_x_final, y, test_size=0.2, shuffle=True)

    # 模型的建立
    autoEncoder, encoder = build_autoEncoder(X_train.shape[1], X_train.shape[2])

    # autoencoder 的训练
    adam = Adam(lr=0.0001)
    sgd = SGD(lr=0.01, momentum=0.7)
    autoEncoder.compile(optimizer=sgd, loss='mse')
    # print(autoEncoder.summary())
    history = autoEncoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=64)

    # 通过encoder提取的特征
    output_encoder_train = encoder.predict(X_train)
    output_encoder_test = encoder.predict(X_test)
    # print('特征维度')
    # print(output_encoder_train.shape)

    # 将特征与原始数据堆叠
    data_concat_train = np.concatenate([X_train, output_encoder_train], axis=2)
    data_concat_test = np.concatenate([X_test, output_encoder_test], axis=2)
    # print('堆叠后维度')
    # print(data_concat_train.shape)

    # 将特征提取出来作为下一个预测网络的输入
    # lstm 预5
    lstm = build_lstm(data_concat_train.shape[1], data_concat_train.shape[2])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_lstm = lstm.fit(data_concat_train, y_train, validation_split=0.2, epochs=1, batch_size=64)

    # cnn 预测
    cnn = build_cnn(data_concat_train.shape[1], data_concat_train.shape[2])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_cnn = cnn.fit(data_concat_train, y_train, validation_split=0.2, epochs=50, batch_size=64)

    # 测试集的预测
    y_pred_proba_lstm = lstm.predict(data_concat_test)
    score_ = []

    y_pred_proba_cnn = cnn.predict(data_concat_test)

    y_pred_cnn = trans_sign(y_pred_proba_cnn, 0.5)

    socre = accuracy_score(y_test, y_pred_cnn)

    recalls = recall_score(y_test,y_pred_cnn)

    pres = precision_score(y_test,y_pred_cnn)

    f1s = f1_score(y_test,y_pred_cnn)

    score_.append(socre)

    keras.backend.clear_session()

    socre_mean = np.mean(np.array(score_))

    print('path={},accuracy cnn:{},recall:{}'.format(path, socre_mean,recalls))


    return socre_mean,recalls,pres,f1s

def main(path,result_score,result_pic):
    # 获取数据
    data_x_final, y = get_data(path)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(data_x_final, y, test_size=0.2, shuffle=True)

    # 模型的建立
    autoEncoder, encoder = build_autoEncoder(X_train.shape[1], X_train.shape[2])

    # autoencoder 的训练
    adam = Adam(lr=0.0001)
    sgd = SGD(lr=0.01, momentum=0.7)
    autoEncoder.compile(optimizer=sgd, loss='mse')
    # print(autoEncoder.summary())
    history = autoEncoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=64)

    # 通过encoder提取的特征
    output_encoder_train = encoder.predict(X_train)
    output_encoder_test = encoder.predict(X_test)
    # print('特征维度')
    # print(output_encoder_train.shape)

    # 将特征与原始数据堆叠
    data_concat_train = np.concatenate([X_train, output_encoder_train], axis=2)
    data_concat_test = np.concatenate([X_test, output_encoder_test], axis=2)
    # print('堆叠后维度')
    # print(data_concat_train.shape)

    # 将特征提取出来作为下一个预测网络的输入
    # lstm 预5
    lstm = build_lstm(data_concat_train.shape[1], data_concat_train.shape[2])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_lstm = lstm.fit(data_concat_train, y_train, validation_split=0.2, epochs=1, batch_size=64)

    # cnn 预测
    cnn = build_cnn(data_concat_train.shape[1], data_concat_train.shape[2])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_cnn = cnn.fit(data_concat_train, y_train, validation_split=0.2, epochs=50, batch_size=64)

    # 测试集的预测
    y_pred_proba_lstm = lstm.predict(data_concat_test)
    score_ = []

    y_pred_proba_cnn = cnn.predict(data_concat_test)
    y_pred_cnn = trans_sign(y_pred_proba_cnn, 0.5)
    socre = accuracy_score(y_test, y_pred_cnn)
    recalls = recall_score(y_test,y_pred_cnn)
    pres = precision_score(y_test,y_pred_cnn)
    f1s = f1_score(y_test,y_pred_cnn)
    score_.append(socre)

    keras.backend.clear_session()
    save = pd.DataFrame(columns=['prob', 'label'], index=range(len(y_test)))
    save['prob'] = np.squeeze(y_pred_proba_cnn)
    save['label'] = y_test

    ori_score = []
    sc_score = []
    for index, row in save.iterrows():
        if row['label'] == 0.0:
            ori_score.append(row['prob'])
        else:
            sc_score.append(row['prob'])

    stop = min(len(sc_score), len(ori_score))
    sc_score = sc_score[:stop]
    ori_score = ori_score[:stop]
    ss = pd.DataFrame(columns=['p_sc', 'label_sc', 'p_ori', 'lael_ori'], index=range(stop))
    ss['p_sc'] = sc_score
    ss['label_sc'] = np.zeros((len(sc_score), 1))
    ss['p_ori'] = ori_score
    ss['label_ori'] = np.ones((len(ori_score), 1))

    filename = path[path.rfind('\\') + 1: len(path) - 4]
    # ff = r'D:\Ho Bae\final_score\rep\\' + filename + '.csv'
    ff = file_result_score + '\\' + filename + '.csv'
    ss.to_csv(ff)

    pic_length = 100
    random_index_sc = random.sample(range(0,len(sc_score)),pic_length)
    random_index_or = random.sample(range(0, len(ori_score)), pic_length)
    random_sc = [sc_score[i] for i in random_index_sc]
    random_or = [ori_score[i] for i in random_index_or]

    plt.plot(range(pic_length), random_sc, c='r', label='sc')
    plt.plot(range(pic_length), random_or, c='blue', label='ori')
    # ff1 = r'C:\Users\Administrator\Desktop\epoch\660_pic\\' + filename + '.jpg'
    ff1 = file_pic_result + '\\' + filename + '.jpg'
    plt.savefig(ff1)
    plt.clf()
    # save.to_csv('score_LSB.csv')

    # print(y_pred_proba_lstm)
    # y_pred_lstm = np.argmax(y_pred_proba_lstm, axis=1)
    y_pred_lstm = trans_sign(y_pred_proba_lstm, 0.5)

    # print(y_pred_proba_cnn)
    # y_pred_cnn = trans_sign(y_pred_proba_cnn, 0.5)
    # y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1)
    # print(y_pred_cnn)

    # 打印结果
    # print('accuracy lstm:')
    # print(accuracy_score(y_test, y_pred_lstm))
    socre_mean = np.mean(np.array(score_))
    # print('path={},accuracy cnn:{}'.format(path,accuracy_score(y_test, y_pred_cnn)))
    print('path={},accuracy cnn:{},recall:{}'.format(path, socre_mean,recalls))
    # print(accuracy_score(y_test, y_pred_cnn))
    # socre = accuracy_score(y_test, y_pred_cnn)

    return socre_mean,recalls,pres,f1s


if __name__ == '__main__':
    modes =['660','888','847']

    for mode in modes:
        # file_ori = r'C:\Users\Administrator\Desktop\epoch\Loss_minium_Ep=53_660' # 生成文件路径 文件格式：txt
        file_ori = r"D:\Desktop\seqs\baseline\{}".format(mode)
        file_csv = r'D:\Desktop\seqs\baseline\{}_csv'.format(mode)  # csv文件路径，原文件与生成文件1：1
        # ori      = r'D:\Desktop\seqs\660\read_in_20\OriginalData\read_20.txt'  # 原文件路径

        file_result_score = r'D:\Desktop\seqs\660\CLASSFINAL\S'  # 单个csv文件的得分
        file_pic_result = r'D:\Desktop\seqs\660\CLASSFINAL\P'  # csv文件中的得分图

        file_finaly = r'D:\Desktop\seqs\baseline\{}.csv'.format(mode)  # 总csv文件的得分值集合

        if mode == '888':
            sample_num = 4500
        elif mode == '847':
            sample_num = 4200  # 生成文件的语句数目，660：3300，888：4500
        else:
            sample_num = 3300
        PATH = []
        for root, dirs, files in os.walk(file_ori):
            for file in files:
                PATH.append(os.path.join(root, file))
        # PATH = PathSlect(PATH)
        BPN = {}
        VAR = {}

        for p in PATH:
            p_ = p[p.rfind('\\') + 1:  len(p) - 4]
            BPN[p_] = cg_tm_kl.find_bpn(p)
            VAR[p_] = cg_tm_kl.find_var(p)
            # if p_.find('fxy6') > 0 or p_.find('fxy3') or p_.find('fxy33') or p_.find('fxy66') > 0:
            #    SeqLength = 198
            #    ori = r'D:\Desktop\seqs\{}\read_3\OriginalData\read_3.txt'.format(mode)
            # else:
            #    SeqLength = 200
            #    ori = r'D:\Desktop\seqs\{}\read_2\OriginalData\read_2.txt'.format(mode)
            SeqLength = 198
            ori = r'D:\Desktop\seqs\{}\read_3\OriginalData\read_3.txt'.format(mode)
            csv_shengc(p, file_csv, ori, sample_num, Seqlength=SeqLength)

        PATH_csv = []
        for root, dirs, files in os.walk(file_csv):
            for file in files:
                PATH_csv.append(os.path.join(root, file))

        temp = pd.DataFrame(columns=['path', 'accurcy', 'reacll_score', 'pres', 'f1s', 'bpn', 'ebpn', 'var'])

        for p in PATH_csv:
            bpn = 0
            ebpn = 0
            var = 0
            p_ = p[p.rfind('\\') + 1:  len(p) - 4]
            accurcy, reaclls, pres, f1s = main(p, result_score=file_result_score, result_pic=file_pic_result)
            bpn = float(BPN[p_])
            ebpn = (1 - accurcy) * bpn * 2
            var = float(VAR[p_])
            temp = temp.append([{'path': p_, 'accurcy': accurcy, 'reacll_score': reaclls, 'pres': pres, 'f1s': f1s,
                                 'bpn': bpn, 'ebpn': ebpn, 'var': var}], ignore_index=True)
            print(temp)

        print(temp)
        temp.to_csv(file_finaly)

        AdgScoreAcc, AriScoreAcc = [], []
        AdgScoreRec, AriScoreRec = [], []
        AdgScorePre, AriScorePre = [], []
        AdgScoreF1s, AriScoreF1s = [], []
        for index, row in temp.iterrows():
            t = row['path']
            num = row['path'].find('adg')
            if row['path'].find('adg') >= 0:
                AdgScoreAcc.append(float(row['accurcy']))
                AdgScoreRec.append(float(row['reacll_score']))
                AdgScorePre.append(float(row['pres']))
                AdgScoreF1s.append(float(row['f1s']))
            else:
                AriScoreAcc.append(float(row['accurcy']))
                AriScoreRec.append(float(row['reacll_score']))
                AriScorePre.append(float(row['pres']))
                AriScoreF1s.append(float(row['f1s']))
        print('AdgAcc:{} AdgReS:{}'.format(np.mean(np.array(AdgScoreAcc)), np.mean(np.array(AdgScoreRec))))
        print('------')
        print('AriAcc:{} AriReS:{}'.format(np.mean(np.array(AriScoreAcc)), np.mean(np.array(AriScoreRec))))

        '''
        # 获取数据
        data_x_final, y = get_data()

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(data_x_final, y, test_size=0.2, shuffle=True)

        # 模型的建立
        autoEncoder, encoder = build_autoEncoder(X_train.shape[1], X_train.shape[2])

        # autoencoder 的训练
        adam = Adam(lr=0.0001)
        sgd = SGD(lr=0.01, momentum=0.7)
        autoEncoder.compile(optimizer=sgd, loss='mse')
        # print(autoEncoder.summary())
        history = autoEncoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=64)

        # 通过encoder提取的特征
        output_encoder_train = encoder.predict(X_train)
        output_encoder_test = encoder.predict(X_test)
        # print('特征维度')
        # print(output_encoder_train.shape)

        # 将特征与原始数据堆叠
        data_concat_train = np.concatenate([X_train, output_encoder_train], axis=2)
        data_concat_test = np.concatenate([X_test, output_encoder_test], axis=2)
        # print('堆叠后维度')
        # print(data_concat_train.shape)

        # 将特征提取出来作为下一个预测网络的输入
        # lstm 预测
        lstm = build_lstm(data_concat_train.shape[1], data_concat_train.shape[2])
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history_lstm = lstm.fit(data_concat_train, y_train, validation_split=0.2, epochs=1, batch_size=64)

        # cnn 预测
        cnn = build_cnn(data_concat_train.shape[1], data_concat_train.shape[2])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history_cnn = cnn.fit(data_concat_train, y_train, validation_split=0.2, epochs=20, batch_size=64)

        # 测试集的预测
        y_pred_proba_lstm = lstm.predict(data_concat_test)
        y_pred_proba_cnn = cnn.predict(data_concat_test)
        # print(y_pred_proba_lstm)
        #y_pred_lstm = np.argmax(y_pred_proba_lstm, axis=1)
        y_pred_lstm = trans_sign(y_pred_proba_lstm, 0.5)

        # print(y_pred_proba_cnn)
        y_pred_cnn = trans_sign(y_pred_proba_cnn, 0.5)
        # y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1)
        # print(y_pred_cnn)



        # 打印结果
        # print('accuracy lstm:')
        # print(accuracy_score(y_test, y_pred_lstm))
        print('accuracy cnn:')
        print(accuracy_score(y_test, y_pred_cnn))

        save = pd.DataFrame(columns=['prob', 'label'], index=range(len(y_test)))
        save['prob'] = np.squeeze(y_pred_proba_cnn)
        save['label'] = y_test

        ori_score = []
        sc_score = []
        for index,row in save.iterrows():
            if row['label'] == 0.0:
                ori_score.append(row['prob'])
            else:
                sc_score.append(row['prob'])

        save.to_csv('score_LSB.csv')

        plt.figure(1)
        plt.plot(history.history['loss'], c='r', label='loss')
        plt.plot(history.history['val_loss'], c='b', label='val_loss')
        plt.legend()

        plt.figure(2)
        plt.plot(history_lstm.history['loss'], c='r', label='loss')
        plt.plot(history_lstm.history['val_loss'], c='b', label='val_loss')
        plt.legend()
        plt.title('LSTM')

        plt.figure(3)
        plt.plot(history_cnn.history['loss'], c='r', label='loss')
        plt.plot(history_cnn.history['val_loss'], c='b', label='val_loss')
        plt.legend()
        plt.title('CNN')

        plt.show()
        '''
