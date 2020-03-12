import numpy as np
from keras.models import model_from_json
from keras.models import Sequential  # Initialise our neural network model as a sequential network
from keras.layers import Conv2D  # Convolution operation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Dropout  # Prevents overfitting by randomly converting few outputs to zero
from keras.layers import MaxPooling2D  # Maxpooling function
from keras.layers import Flatten # Converting 2D arrays into a 1D linear vector
from keras.layers import Dense # Regular fully connected neural network
from keras.preprocessing import image
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
        check_pointer = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=1, save_best_only=True)

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


def load_model_func():
    # load json and create model
    json_file = open('model_mouth_updated.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('weights_mouth.h5')
    print("Loaded model from disk")
    print(loaded_model.summary())
    return loaded_model


def pred(img, loaded_model):
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    single_image = np.vstack([image_array])
    prediction_class = loaded_model.predict(single_image)
    return prediction_class


def load_data(dataset_path):
    data = []
    test_data = []
    test_labels = []
    labels = []

    with open(dataset_path, 'r') as file:
        for line_no, line in enumerate(file.readlines()):
            if 0 < line_no <= NUMBER_OF_IMAGES:
                curr_class, line, set_type = line.split(',')
                image_data = np.asarray([int(line) for line in line.split()]).reshape(15, 15)  # Creating a list out of the string then converting it into a 2-Dimensional numpy array.
                image_data = image_data.astype(np.uint8) / 255.0

                if set_type.strip() == 'PrivateTest':
                    test_data.append(image_data)
                    test_labels.append(curr_class)
                else:
                    data.append(image_data)
                    labels.append(curr_class)

        test_data = np.expand_dims(test_data, -1)
        #test_labels = [1 if num == HAPPY else 0 for num in test_labels]
        data = np.expand_dims(data, -1)
        #labels = [1 if num == HAPPY else 0 for num in labels]

        return np.array(data), np.array(labels), np.array(test_data), np.array(test_labels)


if __name__ == "__main__":
    dataset_path = "newFer2013.csv"
    train_data, train_labels, test_data, test_labels = load_data(dataset_path)

    print("Number of images in Training set:", len(train_data))
    print("Number of images in Test set:", len(test_data))

    # ######HYPERPARAMATERS###########
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    # ################################
    the_model = Model()
    the_model.create_model()
    the_model.compile_model(learning_rate)
    the_model.fit_model(train_data, train_labels, epochs, batch_size)
