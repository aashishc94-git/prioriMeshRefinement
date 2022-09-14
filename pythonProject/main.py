# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from os import mkdir

import tensorflow as tf
import keras.api._v2.keras as keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import plotly.graph_objects as go
from keras.utils import plot_model
from plotly.subplots import make_subplots
from keras.callbacks import TensorBoard
from datetime import datetime

# The function to be traced.
@tf.function
def my_func(x, y):
  # A simple hand-rolled layer.
  return tf.nn.relu(tf.matmul(x, y))

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

img_height = 64
img_width = 64
batch_size = 20
class_names = np.array(['categoryA', 'categoryB', 'categoryC'])

model = keras.Sequential([
    keras.Input((64, 64, 1)),
    layers.Conv2D(32, 3, padding='same', activation='sigmoid'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3),
])
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


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def augment(x, y):
        image = tf.image.random_brightness(x, 0.05)
        return image, y


    ds_train = ds_train.map(augment)

    # for epochs in range(50):
    #     for x, y in ds_train:
    #         # train here
    #         pass


# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

tf.summary.trace_on(graph=True, profiler=True)

z = my_func(x, y)
print(model.summary())
top_layer = model.layers[0]
plt.imshow(top_layer.get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')
plt.show()

# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#           ],
#     metrics=["accuracy"],
# )
#
# history = model.fit(ds_train, batch_size=64, epochs=3, verbose=2, validation_data=ds_validation,
#                     callbacks=[tensorboard_callback])
# model.evaluate(ds_validation, batch_size=64, verbose=2)

# model.save('saved_model/model_1')

# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
