import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from PIL import Image
from sklearn.metrics import accuracy_score


def val_to_key(dict, val):
  '''
  Arguments:
  1) dict: a dictionary <key: value>
           with fruit names as the key and numerical labels as the value
  2) val: a numerical label to be converted to 'descriptive'
  Return:
  The key out of the <key: value> pair.
  '''
  return list(dict.keys())[val]


def convert_file_names(pokedex, max_pkmn):
    dir = "C:/Users/handw/Desktop/test/pokemon-organized"
    for pkmn in pokedex.all_f_names()[:max_pkmn]:
        pkmn_dir = dir + '/' + pkmn
        counter = 0
        for img in sorted(os.listdir(pkmn_dir)):
            pkmn_img_file = pkmn_dir + '/' + img
            os.rename(pkmn_img_file, pkmn_dir + '/' + str(counter) + pkmn_img_file[-4:])
            print(pkmn_img_file)
            counter += 1

##### EDA #####


def show_fact(model, ml):
    images, labels = model.test_gen.next()
    label_dict = model.test_gen.class_indices

    fig, axes = plt.subplots(4, 6)

    for x in range(len(axes)):
        for y in range(len(axes[x])):
            axes[x, y].set_title("Fact: " + val_to_key(label_dict, labels[len(axes[x]) * x + y].argmax()))
            axes[x, y].imshow(images[len(axes[x]) * x + y])

    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


def show_fact_pred(model, ml):
    images, labels = model.test_gen.next()
    label_dict = model.test_gen.class_indices
    y_pred = ml.predict(images)

    fig, axes = plt.subplots(4, 6)
    plt.suptitle('Batch Accuracy: %f\nPrediction | Fact' % accuracy_score(labels.argmax(axis=1), y_pred.argmax(axis=1)))

    for x in range(len(axes)):
        for y in range(len(axes[x])):
            axes[x, y].set_title(val_to_key(label_dict, y_pred[len(axes[x]) * x + y].argmax())
                                 + " | " + val_to_key(label_dict, labels[len(axes[x]) * x + y].argmax()))
            axes[x, y].imshow(images[len(axes[x]) * x + y])

    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


def get_image_counts(pokedex, max_pkmn):
    dir = "C:/Users/handw/Desktop/pkmn"
    pkmn_array = pokedex.all_f_names()[:max_pkmn]
    count_array = []
    for pkmn in pkmn_array:
        pkmn_dir = dir + '/' + pkmn
        counter = 0
        for img in sorted(os.listdir(pkmn_dir)):
            counter += 1

        count_array.append(counter)

    return pkmn_array, count_array


def plot_image_counts(pokedex, max_pkmn):
    pkmn_array, count_array = get_image_counts(pokedex, max_pkmn)
    sns.barplot(x=pkmn_array, y=count_array)


### END OF EDA ###


# Convert an image to a jpeg
def convert_to_jpg(img_path):
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
def resize_img(img, shape):
    img = cv2.resize(img, (shape[0], shape[1]))
    return img


# Normalize pixel values from 0 to 1, important when utilizing NNs
def normalize_img(img):
    img = img / 255.0

    # Does -1 to 1 (unused)
    #img = img / 127.5 - 1
    return img


def remove_transparency(img_path, shape):
    img = open(img_path)
    y, x = np.nonzero(img[:, :, 3])  # get the nonzero alpha coordinates
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)

    crop = (minx, miny, maxx, maxy)
    return open_convert(img_path, shape, crop)


def open(img_path):
    img = Image.open(img_path)
    img = np.array(img)

    return img


# Open an image, convert to jpeg, resize if needed
def open_convert(img_path, shape, crop=None):
    # png
    if img_path[-4:] == '.png':
        img = convert_to_jpg(img_path)
    # jpeg, etc.
    else:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)

    # Crop off excess transparency from img
    if crop is not None:
        img = img[crop[1]:crop[3], crop[0]:crop[2], :]
        Image.fromarray(img).save(img_path)
        #cv2.imwrite(img_path, img)

    # Convert to shape[0]xshape[1]
    img = resize_img(img, shape)

    # Normalize img
    img = normalize_img(img)

    # Return resized img
    return img


# Returns a list of all pokedex images
def load_pokedex_images(pokedex_dir, pokedex, max_pkmn):
    images = []

    for pkmn in pokedex.all_f_names()[:max_pkmn]:
        pkmn += '.jpg'
        if pkmn not in os.listdir(pokedex_dir):
            print("ERROR: Missing image folder for: " + pkmn)
            exit(1)
        images.append(open(os.path.join(pokedex_dir, pkmn)))

    return images
