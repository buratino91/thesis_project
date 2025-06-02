import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import os
import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'Database/basic/Image/aligned/train')
test_dir = os.path.join(base_dir, 'Database/basic/Image/aligned/test')

model = keras.models.load_model('custom_vgg19.keras')

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
class_names = train_ds.class_names

val_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

def preprocess_input(image, label):
    image = keras.applications.vgg19.preprocess_input(image)
    return image, label

preprocessed_ds_train = train_ds.map(preprocess_input)
preprocessed_ds_val = val_ds.map(preprocess_input)
preprocessed_ds_test = test_ds.map(preprocess_input)

AUTOTUNE = tf.data.AUTOTUNE

preprocessed_ds_train = preprocessed_ds_train.cache().prefetch(buffer_size=AUTOTUNE)
preprocessed_ds_val = preprocessed_ds_val.cache().prefetch(buffer_size=AUTOTUNE)
preprocessed_ds_test = preprocessed_ds_test.cache().prefetch(buffer_size=AUTOTUNE)


# base_model = keras.applications.VGG19(
#   include_top=False,
#   weights='imagenet',
#   input_shape=(img_height, img_width, 3)
# )
# # Freeze base model
# base_model.trainable = False

# model = keras.Sequential([
#     base_model,
#     keras.layers.Rescaling(1./255),
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(7, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(
#     preprocessed_ds_train,
#     epochs=30,
#     validation_data=preprocessed_ds_val
# )

# test_loss, test_acc = model.evaluate(val_ds)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")

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
predictions = model.predict(preprocessed_ds_test)
predicted_classes = np.argmax(predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)

cm = confusion_matrix(true_labels, predicted_classes)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=test_ds.class_names, yticklabels=test_ds.class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# model.save('custom_vgg19.keras')
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


