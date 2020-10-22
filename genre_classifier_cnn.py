import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_path = 'data_10.json'


def load_data(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    X = np.array(data['mfcc'])
    y = np.array(data['labels'])

    return X, y


def prepare_datasets(test_size, validation_size):

    X, y = load_data(data_path)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    # test/validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, X_validation, y_train, y_test, y_validation


def plot_history(history):
    _, ax = plt.subplots(2)

    # create accuracy subplots
    ax[0].plot(history.history['accuracy'], label='train accuracy')
    ax[0].plot(history.history['val_accuracy'], label='test accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(loc='lower right')
    ax[0].set_title('Accuracy eval')

    ax[1].plot(history.history['loss'], label='train loss')
    ax[1].plot(history.history['val_loss'], label='test loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Loss eval')

    plt.show()


def build_model(input_shape):

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(
        32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    predict_index = np.argmax(prediction, axis=1)

    print("Expected index: {}, Predicted index: {}".format(y, predict_index))


if __name__ == '__main__':

    X_train, X_test, X_validation, y_train, y_test, y_validation = prepare_datasets(
        0.25, 0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model = build_model(input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(
        X_validation, y_validation), batch_size=16, epochs=30)

    plot_history(history)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print('Accuracy on test set is: {}'.format(test_accuracy))
    print('Loss on test set is: {}'.format(test_error))

    model.save('model.h5')

    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)
