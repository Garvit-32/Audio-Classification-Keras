import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


dataset_path = 'data_10.json'


def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert list into numpy array
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets


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


if __name__ == '__main__':
    # load data
    inputs, targets = load_data(dataset_path)
    # print(inputs.shape)

    # split data into train and test split
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3)

    # build the network architecture
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    history = model.fit(inputs_train, targets_train, validation_data=(
        inputs_test, targets_test), epochs=50, batch_size=32)

    plot_history(history)

    model.save("model.h5")
