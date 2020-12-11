import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout


class VGG16_IN:
    def __init__(self, train_gen, test_gen, epochs=50, batch_size=50):
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_gen = train_gen
        self.test_gen = test_gen

    def train(self):
        # Train VGG16_IN ontop of imagenet
        prior = tf.keras.applications.vgg16.VGG16(
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
        model.add(Dense(self.train_gen.num_classes, activation='softmax', name='Output'))

        # Freeze the VGG16_IN models, e.g. do not train any of its weights.
        # We will just use it as-is.
        for cnn_block_layer in model.layers[0].layers:
            cnn_block_layer.trainable = False
        model.layers[0].trainable = False

        # Compile model
        model.compile(
            optimizer=Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # List a summary of the model
        model.summary()

        # Create callbacks to save top performing model epochs
        callbacks = [
            ModelCheckpoint(
                filepath="models/saved/VGG16_IN/model_epoch_{epoch}",
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1
            )
        ]

        # Fit and train the model
        hist = model.fit(
            self.train_gen,
            steps_per_epoch=self.train_gen.samples // self.batch_size,
            validation_data=self.test_gen,
            validation_steps=self.test_gen.samples // self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks)

        # Plot the model into history graph
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
