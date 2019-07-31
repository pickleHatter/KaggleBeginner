from digitRecognizerHelper import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB


digitRecognizer = [
    ("knn", KNeighborsClassifier()),
    ("svm", SVC(C=5)),
    ("GaussianNB", GaussianNB()),
    ("MultinomialNB", MultinomialNB(alpha=0.1))
]

if __name__ == '__main__':
    train_label, train_data = load_train_data()
    test_data = load_test_data()
    for name, algorithm in digitRecognizer:
        print('digit recognizer using '+name)
        algorithm.fit(train_data, np.ravel(train_label))
        testLabel = algorithm.predict(test_data)
        resultName = './result/'+name+'_Result.csv'
        saveCSV(testLabel, resultName)
