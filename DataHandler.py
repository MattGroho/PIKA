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
