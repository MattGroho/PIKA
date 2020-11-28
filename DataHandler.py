import numpy as np
import pandas as pd
import os

import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Allow very large images to load
Image.MAX_IMAGE_PIXELS = 933120000


class DataHandler:
    def __init__(self, max_pkmn, max_imgs, shape):

        self.max_pkmn = max_pkmn
        self.max_imgs = max_imgs
        self.shape = shape

    def prep_data(self, pokedex):
        # Main directory for pokemon dataset
        pokemon_dir = "/Users/handw/Desktop/testpokemon"

        images = []
        labels = []

        pkmn_count = 0
        count = 0

        for pkmn in pokedex.all_f_names()[:self.max_pkmn]:
            if pkmn not in os.listdir(pokemon_dir):
                print("ERROR: Missing image folder for: " + pkmn)
                exit(1)
            pkmn_count += 1
            if pkmn_count >= self.max_pkmn:
                break
            pkmn_dir = os.path.join(pokemon_dir, pkmn)

            # Current number of images loaded for this pokemon
            curr_imgs = 0

            # Add each image to the list, use most relevant search results
            for img in sorted(os.listdir(pkmn_dir)):
                # Attempt to add image and label to list
                try:
                    # Convert image with image processing and append
                    images.append(self.open_convert(os.path.join(pkmn_dir, img)))
                    # Append label
                    labels.append(pokedex.f_names.index(pkmn))
                # Ignore garbage images
                except (ValueError, OSError):

                    print(os.path.join(pkmn_dir, img) + " failed to load, removing NaN image")
                    os.remove(os.path.join(pkmn_dir, img))
                    continue
                count += 1

                # Increment num images loaded
                curr_imgs += 1
                if curr_imgs >= self.max_imgs:
                    break

        return np.array(images), np.array(labels)
        # Test to display an image / label
        #print(labels[1000])
        #plt.imshow(images[1000])
        #plt.show()

    # Convert an image to a jpeg
    def convert_to_jpg(self, img_path):
        # Convert png to jpeg
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img.load()
            background = Image.new("RGB", img.size, (0, 0, 0))
            background.paste(img, mask=img.split()[3])
            img = np.array(background)
        else:
            img = img.convert('RGB')
            img = np.array(img)

        return img

    # Resize image to shape[0]xshape[1]
    def resize_img(self, img):
        img = cv2.resize(img, (self.shape[0], self.shape[1]))
        return img

    # Normalize pixel values from 0 to 1, important when utilizing NNs
    def normalize_img(self, img):
        img = img / 255.0

        # Does -1 to 1 (unused)
        #img = img / 127.5 - 1
        return img

    # Open an image, convert to jpeg, resize if needed
    def open_convert(self, img_path):
        # png
        if img_path[-4:] == '.png':
            img = self.convert_to_jpg(img_path)
        # jpeg, etc.
        else:
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = np.array(img)

        # Convert to 224x224
        img = self.resize_img(img)

        # Normalize img
        img = self.normalize_img(img)

        # Return resized img
        return img
