import numpy as np
import pandas as pd
import csv
def load_train_data():
    data = []
    with open('./data/train.csv') as file:
         lines = csv.reader(file)
         for line in lines:
             data.append(line)
    data.remove(data[0])
    data = np.asarray(data)
    label = data[:, 0]
    data = data[:, 1:]
    data = data.astype(int)
    label = label.astype(int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 0:
                data[i, j] = 1
    return label, data


def load_test_data():
    data = []
    with open('./data/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            data.append(line)  # 42001*785
    data.remove(data[0])
    data = np.asarray(data)
    data = data.astype(int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 0:
                data[i, j] = 1
    return data


def saveCSV(test_label, csv_name):
    test_index = np.arange(1, len(test_label)+1)
    test_out = np.zeros((len(test_label), 2))
    test_out[:, 0] = test_index
    test_out[:, 1] = test_label
    test_out = test_out.astype(int)
    save = pd.DataFrame(test_out, columns=['ImageId', 'Label'])
    save.to_csv(csv_name, index=False, header=False)
