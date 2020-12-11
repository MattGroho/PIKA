import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from PIL import Image
from skimage import measure
from sklearn.metrics import accuracy_score

# Define final variable(s)
fix_glare = True    # True if applying glare "fix", false if not
save_updated_images = True  # Whether or not to save preprocessed images for viewing
mask_type = 1   # 1 if using create_mask_v1, 2 if using create_mask_v2 (different approaches to calculating white values)

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


def preprocess_img(img_path, shape):
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

    # Fix glare of image and
    if fix_glare:
        glare = None
        if mask_type == 1:
            glare = create_mask_v1(img)
        elif mask_type == 2:
            glare = create_mask_v2(img)
        else:
            print('An error occurred! Please specify a valid mask_type (1 or 2)')
            exit(1)

        img = cv2.inpaint(img, glare, 5, cv2.INPAINT_NS)

    if save_updated_images:
        Image.fromarray(img).save(img_path)

    # Convert to shape[0]xshape[1]
    img = resize_img(img, shape)

    # Normalize img
    img = normalize_img(img)

    # Return resized img
    return img


def create_mask_v1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh_img = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode(thresh_img, None, iterations=2)
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)
    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh_img, neighbors=8, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    return mask


def create_mask_v2(image):
    image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Load the glared image
    h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))  # split into HSV components

    nonSat = s < 180  # Find all pixels that are not very saturated

    # Slightly decrease the area of the non-satuared pixels by a erosion operation.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nonSat = cv2.erode(nonSat.astype(np.uint8), disk)

    # Set all brightness values, where the pixels are still saturated to 0.
    v2 = v.copy()
    v2[nonSat == 0] = 0

    glare = v2 > 200  # filter out very bright pixels.

    # Slightly increase the area for each pixel
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare = cv2.dilate(glare.astype(np.uint8), disk)

    return glare


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