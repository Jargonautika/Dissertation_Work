import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Recurrent():

    def __init__(self, X, y, X_val, y_val, checkpoint_path, csv_path):

        # X is a NxM matrix with observations (waveforms) on the y-axis; and features on the x-axis (2367,517)
        # y is 1,0 array of target labels 0 and 1 [0, 1, 1, 0, 1]
        # X_val is the same as X but from a set of observations withheld for testing (467,517)
        # y_val is the labels for X_val

        self.data = X
        self.labels = y
        self.test = X_val
        self.test_labels = y_val
        self.checkpoint_path = checkpoint_path
        self.csv_path = csv_path
        self.model = self.train()
        self.step = 4


    def train(self):

        # Define the model
        # https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html # TODO work in progress
        model = keras.Sequential()
        model.add(layers.SimpleRNN(units=32, input_shape=(1,self.step), activation="relu"))
        model.add(layers.Dropout(0.4)) # Add dropout to avoid overfitting
        model.add(layers.Dense(8, activation="relu"))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        # Stop early if training loss rises
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        # CSV Logger for checking later
        csv_callback = tf.keras.callbacks.CSVLogger(filename = self.csv_path, separator = ',', append = False)

        # Save the model weights 
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path, save_weights_only=True, verbose=1)

        # Fit the model
        self.history = model.fit(self.data, self.labels, epochs = 150, batch_size = 64, validation_data = (self.test, self.test_labels), callbacks=[es_callback, csv_callback, cp_callback])# , verbose = 0)

        return model