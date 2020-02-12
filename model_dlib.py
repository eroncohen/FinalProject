import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path)
    data = df.drop(["Emotion"], axis=1)
    labels = df["Emotion"]
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True)
    return x_train, y_train, x_test, y_test


clf = SVC(kernel='linear', probability=True, tol=1e-3)
accur_lin = []
best_acc = 0
for i in range(0, 3):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = load_data("landsmarkMTCNN.csv")
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    if pred_lin > best_acc:
        best_acc = pred_lin
        dump(clf, 'svm_modelMTCNN'+str(i)+'.joblib')
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
    disp = plot_confusion_matrix(clf, prediction_data, prediction_labels, cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs
