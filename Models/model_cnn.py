import numpy as np
from keras.models import Sequential  # Initialise our neural network model as a sequential network
from keras.layers import Conv2D  # Convolution operation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Dropout  # Prevents overfitting by randomly converting few outputs to zero
from keras.layers import MaxPooling2D  # Maxpooling function
from keras.layers import Flatten  # Converting 2D arrays into a 1D linear vector
from keras.layers import Dense  # Regular fully connected neural network
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
NUMBER_OF_IMAGES = 16551
HAPPY = '3'


class Model:

    model = Sequential()

    def create_model(self):
        self.model.add(
            Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

    def model_to_json(self):
        json_string = self.model.to_json()
        return json_string

    def compile_model(self, learning_rate):
        adam = optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    def fit_model(self, train_data, train_labels, epochs, batch_size):
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
        early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, mode='auto')
        check_pointer = ModelCheckpoint('weights_cnn.h5', monitor='val_loss', verbose=1, save_best_only=True)

        self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            # verbose=0,
            callbacks=[lr_reducer, check_pointer, early_stopper]
        )


def load_data(dataset_path):
    data = []
    test_data = []
    test_labels = []
    labels = []

    with open(dataset_path, 'r') as file:
        for line_no, line in enumerate(file.readlines()):
            if 0 < line_no <= NUMBER_OF_IMAGES:
                curr_class, line, set_type = line.split(',')
                image_data = np.asarray([int(line) for line in line.split()]).reshape(16, 16)  # Creating a list out of the string then converting it into a 2-Dimensional numpy array.
                image_data = image_data.astype(np.uint8) / 255.0

                if set_type.strip() == 'PrivateTest':
                    test_data.append(image_data)
                    test_labels.append(curr_class)
                else:
                    data.append(image_data)
                    labels.append(curr_class)

        test_data = np.expand_dims(test_data, -1)
        # test_labels = [1 if num == HAPPY else 0 for num in test_labels]
        data = np.expand_dims(data, -1)
        # labels = [1 if num == HAPPY else 0 for num in labels]

        return np.array(data), np.array(labels), np.array(test_data), np.array(test_labels)