import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

class feedForward():

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


    def train(self):

        # Define the model
        model = Sequential()
        model.add(Dense(12, input_dim = self.data.shape[-1], activation = 'relu'))
        model.add(keras.layers.Dropout(0.4)) # Add dropout to avoid overfitting
        model.add(Dense(8, activation = 'relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(Dense(1, activation = 'sigmoid'))

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
