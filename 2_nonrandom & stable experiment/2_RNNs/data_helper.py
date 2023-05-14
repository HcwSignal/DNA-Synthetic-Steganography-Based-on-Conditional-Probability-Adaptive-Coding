import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# 获取训练测试数据
def get_data(path):
    # file_csv = './data/fxyTLSM.csv'
    file_csv = path
    data = pd.read_csv(file_csv, index_col=0)
    print(data.head(5))
    y = data[['label']].values
    y = y.reshape(-1, )
    y = [int(value) for value in y]
    x = data[['text']].values
    data_x = []
    print(len(x[0][0]))
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i][0])):
            if x[i][0][j] != ' ':
                temp.append(x[i][0][j])
        data_x.append(temp[:200])  # 取200个字符
    data_x = np.array(data_x)
    shape1 = data_x.shape[0]
    shape2 = data_x.shape[1]
    data_x = data_x.reshape(-1, 1)
    lb = LabelEncoder()
    lb.fit(data_x)
    print(lb.classes_)
    data_x = lb.transform(data_x)
    print(data_x[0])
    data_x = data_x.reshape((shape1, shape2))
    print(data_x)
    data_x_final = []
    onehot = OneHotEncoder()
    onehot.fit([[0], [1], [2], [3]])
    for i in range(data_x.shape[0]):
        temp_data = data_x[i].reshape(-1, 1)
        temp_data = onehot.transform(temp_data).toarray()
        data_x_final.append(temp_data)
    data_x_final = np.array(data_x_final)
    print(data_x_final)
    print(y)
    return data_x_final, y

def trans_sign(pred_sign, proba):
    trans_data = []
    for i in range(len(pred_sign)):
        if abs(pred_sign[i][0]) > proba:
            trans_data.append(1)
        else:
            trans_data.append(0)
    return trans_data
