import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import argparse
import sys
from logger import Logger
# from torchsummary import summary
#from trend import ple
import numpy as np
#import KL
import data
import textcnn
import cg_tm_kl
import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x: x.lower() == 'true')
    parser.add_argument("--filename", type=str, default='lsb')
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--stop", type=int, default=100)
    args = parser.parse_args(sys.argv[1:])
    return args


args = get_args()


def main(data_helper):
    # ======================
    # 超参数
    # ======================

    STOP = args.stop

    all_var = locals()
    print()
    '''
    for var in all_var:
        if var != "var_name":
            logger.info("{0:15} ".format(var))
            logger.info(all_var[var])
    print()
    '''
    # ======================
    # 数据
    # ======================

    # ======================
    # 构建模型
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = textcnn.TextCNN(
        vocab_size=data_helper.vocab_size,
        embed_size=EMBED_SIZE,
        filter_num=FILTER_NUM,
        filter_size=FILTER_SIZE,
        class_num=CLASS_NUM,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)
    # 	summary(model, (20,))
    criteration = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    early_stop = 0
    best_acc = 0
    best_reacll = 0
    best_precison = 0
    F1score = 0

    # ======================
    # 训练与测试
    # ======================
    for epoch in range(EPOCH):
        generator_train = data_helper.train_generator(BATCH_SIZE)
        generator_test = data_helper.test_generator(BATCH_SIZE)
        train_loss = []
        train_acc = []
        while True:
            try:
                text, label = generator_train.__next__()
            except:
                break
            optimizer.zero_grad()
            y = model(torch.from_numpy(text).long().to(device))
            loss = criteration(y, torch.from_numpy(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            y = y.cpu().detach().numpy()
            train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        test_loss = []
        test_acc = []
        test_precision = []
        test_recall = []
        while True:
            with torch.no_grad():
                try:
                    text, label = generator_test.__next__()
                except:
                    break
                y = model(torch.from_numpy(text).long().to(device))
                loss = criteration(y, torch.from_numpy(label).long().to(device))
                test_loss.append(loss.item())
                y = y.cpu().numpy()
                test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

                for i in range(len(y)):
                    if np.argmax(y[i]) == 1:
                        test_precision += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
                for i in range(len(y)):
                    if label[i] ==1:
                        test_recall += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        logger.info("epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}"
                    .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))

        if np.mean(test_precision) > best_precison:
            best_precison = np.mean(test_precision)

        if np.mean(test_recall) > best_reacll:
            best_reacll = np.mean(test_recall)

        if np.mean(test_acc) > best_acc:
            best_acc = np.mean(test_acc)
        else:
            early_stop += 1
        if early_stop >= STOP:
            try:
                F1score = float(2 *(best_reacll*best_precison) / (best_reacll+best_precison))
            except:
                F1score = 0
            logger.info('best acc: {:.4f},best recall :{:.4f},best precision:{:.4f},F1score:{:.4f}'.format(best_acc,best_reacll,best_precison,F1score))
            #return best_acc,best_reacll,best_precison,F1score

    # if (epoch + 1) % SAVE_EVERY == 0:
    # 			print('saving parameters')
    # 			os.makedirs('models', exist_ok=True)
    # 			torch.save(model.state_dict(), 'models/textcnn-' + str(epoch) + '.pkl')
    # logger.info('best acc: {:.4f}'.format(best_acc))

    return best_acc,best_reacll,best_precison,F1score


if __name__ == '__main__':
    BATCH_SIZE = 50
    EMBED_SIZE = 350
    FILTER_NUM = 512
    FILTER_SIZE = [3, 4, 5]
    CLASS_NUM = 2
    DROPOUT_RATE = 0.2
    EPOCH = args.epoch
    LEARNING_RATE = 0.01
    SAVE_EVERY = 20
    SL = 3
    modes = ['660','888','847']

    for mode in modes:
        dp_ori = r'D:\Desktop\seqs\{}\read_3\OriginalData\read_3.txt'.format(
            mode)  # 原始对比文本路径，read_2即原始数据
        # dirs_name = r'C:\Users\Administrator\Desktop\seqs\zhuangzai_cnn\read_{}/temp/'.format(FILE)  #装载文件夹,将待分析的文件存入
        dirs_name = r'D:\Desktop\seqs\baseline\{}'.format(mode)
        csv_path = r'D:\Desktop\seqs\baseline\{}\{}.csv'.format(mode, mode)
        if mode == '847':
            length_ = 4200
        elif mode == '888':
            length_ = 4500
        else:
            length_ = 3300

        paths = []
        for root, dirs, files in os.walk(dirs_name):
            for file in files:
                paths.append(os.path.join(root, file))

        with open(dp_ori, 'r', encoding='utf-8') as f:
            raw_pos = cg_tm_kl.txt_process_sc_duo(dp_ori, len_sc=198, beg_sc=0, end_sc=length_, PADDING=False,
                                                  flex=80,
                                                  num1=SL, tiqu=False)

        raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

        ff = pd.DataFrame(columns=[])

        for path in paths:
            file_name = path[path.rfind('\\') + 1:-4]

            log_file = r"D:\Desktop\seqs\baseline\{}_LoggerFile.txt".format(file_name)

            logger = Logger(log_file)

            raw_neg = cg_tm_kl.txt_process_sc_duo(path, len_sc=198, beg_sc=0, end_sc=length_, PADDING=False,
                                                  flex=80,
                                                  num1=SL, tiqu=False)
            raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

            logger.info("len_raw_neg{}".format(len(raw_neg)))

            data_helper = data.DataHelper([raw_pos, raw_neg], use_label=True)

            acc, recall, p, f1 = main(data_helper)

            ff = ff.append({'FileName': file_name, 'acc': acc, 'recall': recall, 'precision': p, 'F1': f1},
                           ignore_index=True)

        ff.to_csv(csv_path, sep=',', index=False, header=True)

        print(ff)




