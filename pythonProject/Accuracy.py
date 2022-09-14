import tensorflow as tf
import keras.api._v2.keras as keras
import numpy as np
from keras import layers
import keras_tuner as kt

img_height = 64
img_width = 64
batch_size = 20
class_names = np.array(['categoryA', 'categoryB', 'categoryC'])

train_img = tf.keras.preprocessing.image_dataset_from_directory(
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

validation_img = tf.keras.preprocessing.image_dataset_from_directory(
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


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.Input((64, 64, 1)))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    learning_rate = hp.Choice('learning', values=[1e-3, 2e-1, 1e-4, 1e-2, 2e-3], ordered=True, default=None, )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


tuner = kt.Hyperband(model_builder,
                     objective='accuracy',
                     max_epochs=30,
                     factor=4,
                     directory='new_dir',
                     project_name='kt_2')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', verbose=1, mode="max", restore_best_weights=True)

tuner.search(train_img, batch_size=64, epochs=50, verbose=2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.get('learning'))

