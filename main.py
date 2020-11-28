from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving.save import load_model

from DataHandler import DataHandler
from helper import show_test
from models.ImageClassifier import ImageClassifier
from models.AlexNet import AlexNet
from PokeDex.PokeDex import PokeDex
from models.ModelLoader import ModelLoader

def main():
    # Define final variables
    num_pkmn = 25
    num_imgs = 200
    data_dir = '/Users/handw/Desktop/pkmn'

    target_size = (277, 277)
    batch_size = 50

    load = True
    pokedex = PokeDex('en')

    # (244, 244)
    images, labels = None, None # DataHandler(num_pkmn, num_imgs, (277, 277)).prep_data(pokedex)
    print("Finished loading image data")

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)  # set validation split

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset='training')  # set as training data

    test_gen = train_datagen.flow_from_directory(
        data_dir,  # same directory as training data
        target_size=target_size,
        batch_size=batch_size,
        subset='validation')  # set as validation data

    model = AlexNet(train_gen, test_gen, epochs=300, batch_size=batch_size)
    print("Finished initializing model")

    if load:
        #ml = ModelLoader('vgg16', None)
        #loaded_model = model.load(ml.load_keras_model())

        ml = load_model('/Users/handw/PycharmProjects/PIKA/models/saved/AlexNet/model_epoch_288')
        loaded_model = model.load(ml)

        show_test(model, ml)

    else:
        ml = ModelLoader('vgg16', model.train())
        #ml.save_keras_model()


main()
