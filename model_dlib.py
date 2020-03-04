import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def scaling(X_train):
    preproc = MinMaxScaler()
    return preproc.fit_transform(X_train)


def load_data(path):
    df = pd.read_csv(path)
    data = df.drop(["emotion"], axis=1)
    labels = df["emotion"]
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True)
    return x_train, y_train, x_test, y_test

'''
clf = SVC(kernel='linear', probability=True, tol=1e-3)
accur_lin = []
best_acc = 0
for i in range(0, 3):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = load_data("mouth_arr.csv")
    training_data = scaling(training_data)
    prediction_data = scaling(prediction_data)
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    if pred_lin > best_acc:
        best_acc = pred_lin
        dump(clf, 'svm_model_mouth'+str(i)+'.joblib')
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
    disp = plot_confusion_matrix(clf, prediction_data, prediction_labels, cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs
'''
# Create linear regression object
regr = LinearRegression()
diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = load_data("mouth_arr.csv")
diabetes_X_train = scaling(diabetes_X_train)
diabetes_X_test = scaling(diabetes_X_test)
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
