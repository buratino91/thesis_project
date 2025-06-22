import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras import regularizers
import pathlib
import os
import keras
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    filename="log_file.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

base_dir = os.getcwd()
train_dir = os.path.join(base_dir, "Database/basic/Image/aligned/train_3_classes")
test_dir = os.path.join(base_dir, "Database/basic/Image/aligned/test_3_classes")
test_FER_dir = os.path.join(base_dir, "Database/basic/Image/aligned/test_FER")

checkpoint_filepath = "checkpoint/mdpi_block5and4_checkpoint.model.keras"

img_height = 124
img_width = 124
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123,
)


val_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    label_mode="categorical",
    labels="inferred",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Class weights
# Get class names and counts from the dataset

# y_train_int = np.concatenate([y.numpy().argmax(axis=1) for x, y in train_ds])
# class_weights = dict(enumerate(
#     compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
# ))

class_names = train_ds.class_names
class_counts = np.bincount(
    np.concatenate([y.numpy().argmax(axis=1) for x, y in train_ds])
)

# Compute class weights
total_samples = sum(class_counts)
class_weights = {
    i: total_samples / (len(class_counts) * count)
    for i, count in enumerate(class_counts)
}


test_ds = keras.utils.image_dataset_from_directory(  # returns (images, label)
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels="inferred",
    seed=123,
    label_mode="categorical",
)

test_classnames = test_ds.class_names


def preprocess_input(image, label):
    image = keras.applications.vgg19.preprocess_input(image)
    return image, label


train_ds = train_ds.map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, mode="min"
)

# Reduce learning rate when metric has stopped improving
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5
)

# save model checkpoint during training
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

# data augmentation
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
    ]
)


base_model = keras.applications.VGG19(
    include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3)
)
# Freeze base model
base_model.trainable = False
for layer in base_model.layers:
    if "block4" in layer.name or "block5" in layer.name:
        layer.trainable = True


model = keras.Sequential(
    [
        base_model,
        data_augmentation,
        keras.layers.Flatten(),
        keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.0085)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, kernel_regularizer=regularizers.l2(0.0085)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(
            3,
            activation="softmax", kernel_regularizer=regularizers.l2(0.0085)
        ),
    ]
)

#model_checkpoint = keras.models.load_model(checkpoint_filepath)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# model.summary()
history = model.fit(
    train_ds,
    epochs=60,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, model_checkpoint_callback],
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Test model on an image
# img = keras.utils.load_img('istock-1351285222-sad-man-wit-tear-lr-jpg.jpg', target_size=(img_height, img_width))
# img_array = keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )

# Test model with dir of images and check confusion matrix
# predictions = model.predict(test_ds)
# predicted_classes = np.argmax(predictions, axis=1)
# true_labels = np.concatenate([y for x, y in test_ds], axis=0)


# cm = confusion_matrix(true_labels, predicted_classes)
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=test_classnames, yticklabels=test_classnames)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()


# #     1: 'Surprise',
# #     2: 'Fear',
# #     3: 'Disgust',
# #     4: 'Happiness',
# #     5: 'Sadness',
# #     6: 'Anger',
# #     7: 'Neutral'
