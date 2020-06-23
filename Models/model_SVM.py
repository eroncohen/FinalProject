import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def scaling(x_train):
    preproc = MinMaxScaler()
    return preproc.fit_transform(x_train)


def load_data(path):
    df = pd.read_csv(path)
    data = df.drop(["Emotion"], axis=1)
    labels = df["Emotion"]
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True)
    return x_train, y_train, x_test, y_test


# clf = SVC(kernel='rbf', probability=True, tol=1e-3)
# clf = LogisticRegression(max_iter=1000)
clf = RandomForestClassifier(n_estimators=300, bootstrap=True, max_features='sqrt')

accur_lin = []
best_acc = 0

for i in range(0, 3):
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = load_data("../data/csv/our_mtcnn_new.csv")
    # training_data = scaling(training_data)
    # prediction_data = scaling(prediction_data)
    npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" % i)  # train SVM
    clf.fit(npar_train, npar_trainlabs)
    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    # print(clf.gamma)
    if pred_lin > best_acc:
        best_acc = pred_lin
        dump(clf, 'svm_model_our_mtcnn_rbf'+str(i)+'.joblib')
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)  # Store accuracy in a list
    disp = plot_confusion_matrix(clf, prediction_data, prediction_labels, cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()
print("Mean value lin svm: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs


