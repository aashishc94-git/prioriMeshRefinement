# import tensorflow as tf
import tensorflow as tf
import tensorflow_datasets as tfds
import keras.api._v2.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

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
x_train = None
for image,label in tfds.as_numpy(ds_train):
    x_train = image


train_images = np.concatenate([x for x, y in ds_train], axis=0)
train_labels = np.concatenate([y for x, y in ds_train], axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(20, drop_remainder=True)

test_images = np.concatenate([x for x, y in ds_validation], axis=0)
test_labels = np.concatenate([y for x, y in ds_validation], axis=0)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(20, drop_remainder=True)

X, y = ds_train, ds_validation

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X, y,
                                                        cv=10, scoring='accuracy', n_jobs=-1,
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
