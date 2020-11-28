from keras_preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import MaxPooling2D, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.np_utils import to_categorical


class ImageClassifier:
    def __init__(self, data, labels, num_pkmn):
        self.num_pkmn = num_pkmn

        # Initialize null data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Propagate data
        self.prep_data(data, to_categorical(labels))

    def prep_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)

    def train(self):
        #train_data = ImageDataGenerator()
        #train_data = train_data.flow_from_directory(directory="/Users/handw/Desktop/testpokemon", target_size=(224, 224))
        #test_data = ImageDataGenerator()
        #test_data = test_data.flow_from_directory(directory="/Users/handw/Desktop/testpokemon", target_size=(224, 224))
        """
        models = Sequential()
        models.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        models.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        models.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        models.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        models.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        models.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        models.add(Flatten())
        models.add(Dense(units=4096, activation="relu"))
        models.add(Dense(units=4096, activation="relu"))
        models.add(Dense(units=self.num_pkmn - 1, activation="softmax"))

        optimizer = Adam(lr=.001)

        models.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])####

        # models.fit(x=self.X_train, y=self.y_train, epochs=10)
        """
        prior = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        model = Sequential()
        model.add(prior)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25, name='Dropout_Regularization'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.75, name='Final_Regularization'))
        model.add(Dense(self.num_pkmn - 1, activation='softmax', name='Output'))

        # Freeze the VGG16 models, e.g. do not train any of its weights.
        # We will just use it as-is.
        for cnn_block_layer in model.layers[0].layers:
            cnn_block_layer.trainable = False
        model.layers[0].trainable = False

        # Compile the models. I found that RMSprop with the default learning
        # weight worked fine.
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        hist = model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test), epochs=50, batch_size=50)

        #print(models.predict(self.X_test[10]))

        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("models accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        plt.show()

        return model

    def load(self, model):
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        score = model.evaluate(self.X_test, self.y_test, verbose=0)

        print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))

        return model
