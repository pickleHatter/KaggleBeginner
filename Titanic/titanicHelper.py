import pandas as pd
import numpy as np
# read data from file


def load_data(train_data=True):
    if train_data:
        data = pd.read_csv('data/train.csv')
    else:
        data = pd.read_csv('data/test.csv')
    print(data.info())
    # fill nan values with 0
    data = data.fillna(0)
    # convert ['male', 'female'] values of Sex to [1, 0]
    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)

    # select features and labels for training
    if train_data:
        info = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
        label = data['Survived']
        return info, label
    else:
        info = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
        return info


def saveCSV(test_label, csv_name):
    test_index = np.arange(892, 892+len(test_label))
    test_out = np.zeros((len(test_label), 2))
    test_out[:, 0] = test_index
    test_out[:, 1] = test_label
    test_out = test_out.astype(int)
    save = pd.DataFrame(test_out, columns=['PassengerId', 'Survived'])
    save.to_csv(csv_name, index=False, header=False)