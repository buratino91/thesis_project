import keras
import os
import tensorflow as tf

base_dir = os.getcwd()
test_dir = os.path.join(base_dir, "Database/basic/Image/aligned/test_3_classes")

checkpoint_filepath = "checkpoint/vgg16_checkpoint.model.keras"
img_height = 124
img_width = 124
batch_size = 32

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

test_ds = test_ds.map(preprocess_input).cache().prefetch(tf.data.AUTOTUNE)

trained_model = keras.models.load_model(checkpoint_filepath)
test_loss, test_acc = trained_model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")