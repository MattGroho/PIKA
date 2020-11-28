from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


class AlexNet:
    def __init__(self, train_gen, test_gen, epochs=50, batch_size=50):
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_gen = train_gen
        self.test_gen = test_gen

    def train(self):
        model = Sequential()
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(277, 277, 3)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.train_gen.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001),
                      metrics=['accuracy'])
        model.summary()

        callbacks = [
            ModelCheckpoint(
                filepath="models/saved/AlexNet/model_epoch_{epoch}",
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1
            )
        ]

        hist = model.fit(
            self.train_gen,
            steps_per_epoch=self.train_gen.samples // self.batch_size,
            validation_data=self.test_gen,
            validation_steps=self.test_gen.samples // self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks)

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
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001),
                      metrics=['accuracy'])

        score = model.evaluate(self.test_gen, verbose=0)

        print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))

        return model
