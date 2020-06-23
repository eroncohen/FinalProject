from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers
from Models.model_cnn import load_data
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

train_data, train_labels, test_data, test_labels = load_data("mouth_only_pic.csv")
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, mode='auto')
check_pointer = ModelCheckpoint('weights_mouth_cnn.h5', monitor='val_loss', verbose=1, save_best_only=True)

model.fit(
    train_data,
    train_labels,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    shuffle=True,
    # verbose=0,
    callbacks=[lr_reducer, check_pointer, early_stopper]
)

json_string = model.to_json()
print(json_string)
with open('Classifiers/model_mouth_cnn.json', 'w') as json_file:
    json_file.write(json_string)
