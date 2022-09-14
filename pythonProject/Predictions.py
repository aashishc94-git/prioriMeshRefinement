import tensorflow as tf
import keras.api._v2.keras as keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

img_height = 64
img_width = 64
batch_size = 20
class_names = np.array(['categoryA', 'categoryB', 'categoryC'])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    'inferred',
    "int",
    ['categoryA', 'categoryB', 'categoryC'],
    'grayscale',
    batch_size,
    (img_height, img_width),
    True,
    123,
    0.1,
    "training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    'inferred',
    "int",
    ['categoryA', 'categoryB', 'categoryC'],
    'grayscale',
    batch_size,
    (img_height, img_width),
    True,
    123,
    0.1,
    "validation"
)

loaded_model = tf.keras.models.load_model('saved_model/model_1')

probability_model = keras.Sequential([loaded_model,
                                      keras.layers.Softmax()])

predictions = probability_model.predict(ds_validation)



img = np.concatenate([x for x, y in ds_validation], axis=0)
labels = np.concatenate([y for x, y in ds_validation], axis=0)

flat_labels = ds_train.iloc[:, 0].values
print(" Image Labels :  ", flat_labels[0:20])

fig = plt.figure()

# this function is used to update the plots for each epoch and error
def plt_dynamic(x, y, y_1, ax, ticks, title, colors=['b']):
    ax.plot(x, y, 'b', label="Train Loss")
    ax.plot(x, y_1, 'r', label="Test Loss")
    if len(x) == 1:
        plt.legend()
        plt.title(title)
    plt.yticks(ticks)
    fig.canvas.draw()


one_hot_encoder = OneHotEncoder(sparse=False)
flat_labels = flat_labels.reshape(len(flat_labels), 1)
labels = one_hot_encoder.fit_transform(flat_labels)
labels = labels.astype(np.uint8)


# print(np.argmax(predictions[32]))
# print(len(labels))
# counter = 0
# for x in range(len(labels)):
#     if labels[x] == np.argmax(predictions[x]):
#         counter = counter+1
#
# print(counter)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(3))
    plt.yticks([])
    thisplot = plt.bar(range(3), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 8
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], labels, img)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], labels)
plt.tight_layout()
plt.show()

# global array
#
# for image, label in ds_validation.take(1):
#     print("Image shape: ", image.numpy().shape)
#     array = label.numpy()
#
# print("Label: ", array)
# length = len(array)
#
# print(length)
# print(predictions[2])
# index = 0
