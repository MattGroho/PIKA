import matplotlib.pyplot as plt
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


def show_test(model, ml):
    images, labels = model.test_gen.next()
    label_dict = model.test_gen.class_indices
    y_pred = ml.predict(images)

    fig, axes = plt.subplots(5, 10)
    plt.suptitle('Batch Accuracy: %f\nPrediction | Fact' % accuracy_score(labels.argmax(axis=1), y_pred.argmax(axis=1)))

    for x in range(len(axes)):
        for y in range(len(axes[x])):
            axes[x, y].set_title(val_to_key(label_dict, y_pred[len(axes[x]) * x + y].argmax())
                                 + " | " + val_to_key(label_dict, labels[len(axes[x]) * x + y].argmax()))
            axes[x, y].imshow(images[len(axes[x]) * x + y])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
