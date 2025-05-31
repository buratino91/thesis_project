import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import os
import keras

base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'Database/basic/Image/aligned/train')

img_height = 224
img_width = 224
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

def preprocess_input(image, label):
    image = keras.applications.vgg19.preprocess_input(image)
    return image, label

preprocessed_ds_train = train_ds.map(preprocess_input)
preprocessed_ds_val = val_ds.map(preprocess_input)

AUTOTUNE = tf.data.AUTOTUNE

preprocessed_ds_train = preprocessed_ds_train.cache().prefetch(buffer_size=AUTOTUNE)
preprocessed_ds_val = preprocessed_ds_val.cache().prefetch(buffer_size=AUTOTUNE)


base_model = keras.applications.VGG19(
  include_top=False,
  weights='imagenet',
  input_shape=(img_height, img_width, 3)
)
# Freeze base model
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.Rescaling(1./255),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    preprocessed_ds_train,
    epochs=10,
    validation_data=preprocessed_ds_val
)

test_loss, test_acc = model.evaluate(val_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Compound labels
# 1: Happily Surprised
# 2: Happily Disgusted
# 3: Sadly Fearful
# 4: Sadly Angry
# 5: Sadly Surprised    
# 6: Sadly Disgusted
# 7: Fearfully Angry
# 8: Fearfully Surprised
# 9: Angrily Surprised
# 10: Angrily Disgusted
# 11: Disgustedly Surprised


# #     1: 'Surprise',
# #     2: 'Fear',
# #     3: 'Disgust',
# #     4: 'Happiness',
# #     5: 'Sadness',
# #     6: 'Anger',
# #     7: 'Neutral'


