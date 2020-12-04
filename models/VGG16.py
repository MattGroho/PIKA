import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import MaxPool2D, Conv2D


class VGG16:
    def __init__(self, train_gen, test_gen, epochs=50, batch_size=50):
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_gen = train_gen
        self.test_gen = test_gen

    def train(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=self.train_gen.num_classes, activation="softmax"))

        model.compile(
            optimizer=Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        callbacks = [
            ModelCheckpoint(
                filepath="models/saved/VGG16/model_epoch_{epoch}",
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

    def evaluate(self, model):
        score = model.evaluate(self.test_gen, verbose=0)

        print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))
