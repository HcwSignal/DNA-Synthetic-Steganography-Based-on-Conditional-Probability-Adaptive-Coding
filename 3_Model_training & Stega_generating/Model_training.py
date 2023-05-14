import torch
from torch import nn
import torch.optim as optim
import scipy.stats
import numpy as np
from logger import Logger
import utils
import lm
import os
import inspect
def continue_training(path):
    File = []

    for root, dirs, files in os.walk(path):
        for file in files:
            File.append(os.path.join(root, file))

    Models = []
    for f in File:
        if f.find('pkl') > 0:
            Models.append(f)

    m_name = []
    epoch_nums = []
    for m in Models:
        m_ = m[m.rfind('\\') + 1: -4]
        m_name = m_.split('-')
        epoch_nums.append(int(m_name[-1]))
        # for elements in m_name:
        #    try:
        #        epoch_nums.append(int(elements))
        #    except:
        #        pattern = elements

    try:
        return_num = str(max(epoch_nums))
    except:
        return_num = 0
    return return_num

def main(file, code, SeqLength, ind,Path_model_save,mode, modelnum,lr,running_mode):
    print(torch.__version__)
    print(torch.cuda.is_available())
    # f_p = open("D:\Destop\prob.txt",'w')
    # ======================
    # 超参数
    # ======================
    CELL = "lstm"  # rnn, gru, lstm
    # DATASET = 'news'
    DATASET = 'base1'
    RATIO = 0.9
    WORD_DROP = 0
    MIN_LEN = SeqLength
    MAX_LEN = 200
    BATCH_SIZE = 30
    EMBED_SIZE = 350
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    START_EPOCH = 0
    if running_mode == 't': #test
        EPOCH = 3
    elif running_mode == 's': #start_train
        EPOCH = 150
    elif  running_mode == 'c': #continue_train
        EPOCH = 50

    LEARNING_RATE = lr
    MAX_GENERATE_LENGTH = SeqLength  # 每条语句长度为200或者为198
    GENERATE_EVERY = 5
    PRINT_EVERY = 1
    SEED = 100
    repeat = 1
    output1 = []  # 用于记录每一次生成的数据
    num1 = 0
    early_stop = 0
    STOP = 100

    all_var = locals()
    print()

    #log_file = r"D:\Desktop\additional data\{}\read_{}\Models_{}".format(mode,ind, modelnum) + '\\{}-{}.txt'.format(code, modelnum)
    log_file = Path_model_save + r"\{}\read_{}\M_{}".format(mode, ind, str(modelnum) )+ '\\{}.txt'.format(code)

    logger = Logger(log_file)
    logger.info(
        'Epoch {} Lr {} BatchSize{} EmbedSize{} HiddenDim{}'.format(EPOCH, LEARNING_RATE, BATCH_SIZE, EMBED_SIZE,
                                                                    HIDDEN_DIM))
    # ======================
    # 数据
    # ======================
    # data_path = '/Codes/Python/__data/dataset2020/' + DATASET + '2020.txt'
    # data_path = '../_data/' + DATASET + '2020.txt'
    data_path = file
    # data_path = r'D:\1\ori_two.txt' #原数据，以读入的方式
    # train_path = 'train_' + DATASET
    train_path = Path_model_save + r"\{}\read_{}\M_{}".format(mode,ind, str(modelnum)) + '\\train.txt'  # train数据，以写入的方式存储训练数据，取原数据的前0.9
    # test_path = 'test_' + DATASET
    test_path =  Path_model_save + r"\{}\read_{}\M_{}".format(mode,ind, str(modelnum)) + '\\test.txt'
    vocabulary = utils.Vocabulary(
        data_path,
        max_len=MAX_LEN,

        min_len=MIN_LEN,
        word_drop=WORD_DROP)

    utils.split_corpus(data_path, train_path, test_path, max_len=MAX_LEN, min_len=MIN_LEN, ratio=RATIO,
                       seed=SEED)  # 分割语料库，将原数据data_path的前0.9存储为trian，后0.1存储为test
    train = utils.Corpus(train_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)  # 构建trian语料库
    test = utils.Corpus(test_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)  # 构建test语料库
    train_generator = utils.Generator(train.corpus, vocabulary=vocabulary)
    test_generator = utils.Generator(test.corpus, vocabulary=vocabulary)

    # ======================
    # 构建模型
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = lm.LM(
        cell=CELL,
        vocab_size=vocabulary.vocab_size,
        embed_size=EMBED_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {:d}".format(total_trainable_params))
    criterion = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
    #                       amsgrad=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=1e-2, amsgrad=False)
    print()

    # ======================
    # 训练与测试
    # ======================
    best_loss = 1000000
    step = 0
    # print('models/' + DATASET + '-' + str(START_EPOCH) + '.pkl')
    print()
    logger.info("------------------------")

    if test == 'c':
        file_load = Path_model_save + r"\{}\read_{}\M_{}".format(mode, ind, str(modelnum))
        LOAD_EPOCH = continue_training(file_load)
    else:
        LOAD_EPOCH = 0

    if LOAD_EPOCH != 0:
        model_load = file_load + '\\fxy{}-{}-{}.pkl'.format(ind,modelnum,LOAD_EPOCH)
        model.load_state_dict(torch.load(model_load, map_location=device)) #加载模型

    for epoch in range(int(LOAD_EPOCH) + 1, EPOCH + int(LOAD_EPOCH)):
        train_g = train_generator.build_generator(BATCH_SIZE)
        test_g = test_generator.build_generator(BATCH_SIZE)
        train_loss = []
        model.train()  #训练模式，保持drop_out
        while True:
            try:
                text = train_g.__next__()
            except:
                break
            optimizer.zero_grad()
            try:
                text = np.array(text).reshape((BATCH_SIZE, -1))
            except:
                print(np.shape(text))
            text_in = text[:, :-1]  # 将原数据text去除最后一列，作为网络输入
            text_target = text[:, 1:]  # 将原数据text去除开头一列，作为target
            y = model(torch.from_numpy(text_in).long().to(device))
            # input和target是同一个text掐头去尾？去掉起始标与结尾标的意义？
            yyy = y.reshape(-1, vocabulary.vocab_size)
            xxx = torch.from_numpy(text_target).reshape(-1).long().to(
                device)
            loss = criterion(yyy, xxx)  # y.shape=[32,101,8],vocabulary.vocab_size=8
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
            step += 1
            torch.cuda.empty_cache()  # 回收显存，清除没用的临时变量

        # if step % PRINT_EVERY == 0:
        # print('step {:d} training loss {:.4f}'.format(step, loss.item()))

        test_loss = []
        model.eval()  # 测试模式，激活所有神经元
        with torch.no_grad():  # 停止梯度计算，加速、节省GPU空间
            while True:
                try:
                    text = test_g.__next__()
                except:
                    break
                text_in = text[:, :-1]
                text_target = text[:, 1:]
                y = model(torch.from_numpy(text_in).long().to(device))
                loss = criterion(y.reshape(-1, vocabulary.vocab_size),
                                 torch.from_numpy(text_target).reshape(-1).long().to(device))
                test_loss.append(loss.item())
                torch.cuda.empty_cache()

        logger.info('epoch {:d}   training loss {:.4f}    test loss {:.4f}'
                    .format(epoch, np.mean(train_loss), np.mean(test_loss)))

        if np.mean(test_loss) < best_loss:  # best_loss设置为一百万,更新best_loss，当test_loss开始变小的时候就保存模型，取最后的模型
            best_loss = np.mean(test_loss)
            print('-----------------------------------------------------')
            print('saving parameters')
            os.makedirs('models', exist_ok=True)

            torch.save(model.state_dict(),
                       Path_model_save + r"\{}\read_{}\M_{}".format(mode,ind, str(modelnum)) + '\\' + code + '-' + str(modelnum) + '-'+ str(epoch) + '.pkl')  # 保存模型，包括权重，参数等等
            print('-----------------------------------------------------')
        else:
            early_stop += 1

        if early_stop >= STOP:
            return 0

        prob = []
        if (epoch + 1) % GENERATE_EVERY == 0:  # 每五轮生成一次文本
            # if (epoch + 1) % 3 == 0:  # 每五轮生成一次文本
            model.eval()
            with torch.no_grad():
                # 生成文本
                x = torch.LongTensor([[vocabulary.w2i['_BOS']]] * repeat).to(device)
                samp = model.sample_beg(x)
                x = torch.cat([x, samp], dim=1)  # 按列进行拼接
                # str1 = x.cpu().numpy()
                for i in range(MAX_GENERATE_LENGTH - 1):
                    samp = model.sample_beg(x)
                    # samp = model.sample_1(x, AA_num, vocabulary)
                    # f_p.writelines(str(prob))
                    # f_p.write("\n")
                    x = torch.cat([x, samp], dim=1)  # 按列进行拼接

                x = x.cpu().numpy()

            print('-----------------------------------------------------')
            for i in range(x.shape[0]):  # shape[0]为x矩阵的行数
                output1 = ' '.join([vocabulary.i2w[_] for _ in list(x[i, :]) if _ not in
                                    [vocabulary.w2i['_BOS'], vocabulary.w2i['_EOS'], vocabulary.w2i['_PAD']]])

                print(' '.join([vocabulary.i2w[_] for _ in list(x[i, :]) if _ not in
                                [vocabulary.w2i['_BOS'], vocabulary.w2i['_EOS'], vocabulary.w2i['_PAD']]]))

            print('-----------------------------------------------------')

    logger.info("{}            {}".format(file, code))

