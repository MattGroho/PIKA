import pandas as pd

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving.save import load_model

from app.MainApp import MainApp
from helper import plot_image_counts, show_fact
from models.VGG16 import VGG16
from models.VGG16_IN import VGG16_IN
from models.AlexNet import AlexNet
from PokeDex.PokeDex import PokeDex


def main():
    # Define final variables
    model_type = 'VGG16_IN'  # Options include AlexNet, VGG16, or VGG16_IN
    train = False  # True if training new model, False if launching application

    # Define workflow paths
    data_dir = '/Users/handw/Desktop/pkmn'
    pokemon_dir = '/Users/handw/PycharmProjects/PIKA/'
    pokedex_dir = pokemon_dir + 'PokeDex/images/'
    saved_model_dir = None

    # Training / testing parameters
    target_size = None
    epochs = 300
    batch_size = 50

    model = None

    # Initialize PokeDex in English (Not used in current build)
    pokedex = PokeDex('en')

    if model_type == 'AlexNet':
        saved_model_dir = pokemon_dir + 'models/saved/AlexNet/model_epoch_264'
        target_size = (277, 277)
    elif model_type == 'VGG16':
        saved_model_dir = pokemon_dir + 'models/saved/VGG16/model_epoch_1'
        target_size = (224, 224)
    elif model_type == 'VGG16_IN':
        saved_model_dir = pokemon_dir + 'models/saved/VGG16_IN/model_epoch_122'
        target_size = (224, 224)

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

    label_dict = train_gen.class_indices

    if model_type == 'AlexNet':
        model = AlexNet(train_gen, test_gen, epochs=epochs, batch_size=batch_size)
    elif model_type == 'VGG16':
        model = VGG16(train_gen, test_gen, epochs=epochs, batch_size=batch_size)
    elif model_type == 'VGG16_IN':
        model = VGG16_IN(train_gen, test_gen, epochs=epochs, batch_size=batch_size)

    print("Finished initializing model")

    if train:
        model.train()
    else:
        loaded_model = load_model(saved_model_dir)
        show_fact(model, loaded_model)

        # Load pokemon info dataset
        #df = pd.read_csv(pokemon_dir + 'PokeDex/Pokemon Info.csv', encoding="ISO-8859-1")

        #app = MainApp(loaded_model, df, pokedex, pokedex_dir, label_dict, train_gen.num_classes, target_size)
        #app.run()


main()
