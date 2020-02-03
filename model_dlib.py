import numpy as np
from keras.models import model_from_json
import keras as k
import pandas as pd
from keras.models import Sequential, load_model  # Initialise our neural network model as a sequential network
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Dropout  # Prevents overfitting by randomly converting few outputs to zero
from keras.layers import Dense # Regular fully connected neural network
from keras.preprocessing import image
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_data(path):
    df = pd.read_csv(path)
    data = df.drop(["Emotion"], axis=1)
    labels = df["Emotion"]
    data_scaler = MinMaxScaler()
    data_scaler.fit(data)
    column_names = data.columns
    data[column_names] = data_scaler.transform(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data("landsmark.csv")
model = Sequential()
model.add(Dense(256, input_dim=64, kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=3)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, mode='auto')
check_pointer = ModelCheckpoint('new_weights.h5', monitor='loss', verbose=1, save_best_only=True)
'''
model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    shuffle=True,
    # verbose=0,
    callbacks=[lr_reducer, check_pointer]#, early_stopper]
)
'''
model.fit(x_train, y_train, epochs=200, batch_size=64)
model.save("savedmodel")


print("Model file:  savedmodel")
model = load_model("savedmodel")
pred = model.predict(x_test)
pred = [1 if y >= 0.5 else 0 for y in pred] #Threshold, transforming probabilities to either 0 or 1 depending if the probability is below or above 0.5
scores = model.evaluate(x_test, y_test)
print()
print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
print()
print("Predicted : {0}".format(", ".join([str(x) for x in pred])))
print()
print("Scores    : loss = ", scores[0], " acc = ", scores[1])
print("---------------------------------------------------------")
print()

