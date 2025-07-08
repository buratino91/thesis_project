import numpy as np
import os
import seaborn as sns
import tensorflow as tf
from keras import regularizers
import pathlib
import os
import keras
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

base_dir = os.getcwd()
train_dir = os.path.join(base_dir, "Database/basic/Image/aligned/train_3_classes")
test_dir = os.path.join(base_dir, "Database/basic/Image/aligned/test_3_classes")
test_FER_dir = os.path.join(base_dir, "Database/basic/Image/aligned/test_FER")

checkpoint_filepath = "checkpoint/vgg16_checkpoint.model.keras"

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


def preprocess_input(image, label):
    image = keras.applications.vgg16.preprocess_input(image)
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
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.2, 0.2),
    ]
)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

base_model = keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
    pooling="max",
)
# Freeze base model
base_model.trainable = False

for layer in base_model.layers:
    if "block5" in layer.name:
        base_model.trainable = True

model = keras.Sequential(
    [
        data_augmentation,
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            3, activation="softmax", kernel_regularizer=regularizers.l2(0.001)
        ),
    ]
)

trained_model = keras.models.load_model(checkpoint_filepath)
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

# # model.summary()
# history = model.fit(
#     train_ds,
#     epochs=60,
#     validation_data=val_ds,
#     class_weight=class_weights,
#     callbacks=[early_stopping, reduce_lr, model_checkpoint_callback, tensorboard_callback],
# )

# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Test model with dir of images and check confusion matrix
predictions = trained_model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
y_true = np.argmax(true_labels, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
