import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from keras import callbacks
import keras_tuner as kt
from keras_tuner import HyperModel 
import os

csv_path_train = os.path.join('Database', 'trainingdata_basic.csv')
csv_path_test = os.path.join('Database', 'testdata_basic.csv')

df_train = pd.read_csv(csv_path_train)
df_test = pd.read_csv(csv_path_test)

df_train['Images'] = df_train['Images'].apply(lambda x: os.path.join('Database', x.split('Database/')[-1]))
df_test['Images'] = df_test['Images'].apply(lambda x: os.path.join('Database', x.split('Database/')[-1]))


# tf.data pipeline
def load_image_and_label(image_path, label):
    # Read the image file
    image = tf.io.read_file(image_path)

    # Convert image to RGB
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize images to 224x224
    image = tf.image.resize(image, [224, 224])

    # Normalize pixel values
    image = image / 255.0

    return image, label

# Train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((df_train['Images'], df_train["Stress?"]))
train_dataset = train_dataset.map(load_image_and_label).batch(32).prefetch(tf.data.AUTOTUNE)

# Test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((df_test['Images'], df_test["Stress?"]))
test_dataset = test_dataset.map(load_image_and_label).batch(32).prefetch(tf.data.AUTOTUNE)

# For tuning
def build_model(hp):
    model = tf.keras.Sequential()
    
    # Tune Conv layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Choice(f'filters_{i}', [32, 64, 128]),
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    
    # Tune Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 2)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'dense_units_{i}', 64, 256, step=64),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dropout(
            hp.Float('dropout_rate', 0.1, 0.5)
        ))
    
    # Binary output
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Tune learning rate
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model   

# Build model from tuned parameters
def build_best_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3))),
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    #Dense layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.42867))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.42867))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0010384),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
    )
    return model

best_model = build_best_model()
# best_model.summary()

# Train the model
# history = best_model.fit(
#     train_dataset,
#     epochs=30,
#     validation_data=test_dataset,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
#     ]
# )

# test_loss, test_acc = best_model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.4f}")
# print(f"Test loss: {test_loss:.4f}")

# tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=30,
#     factor=3,
#     directory='tuner_dir',
#     project_name='stress_cnn'
# 

# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# tuner.search(train_dataset, epochs=50, validation_data=test_dataset, callbacks=[stop_early])

# # Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)
# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.summary()

# model.fit(train_dataset, epochs=10, validation_data=test_dataset)
best_model.save('best_model_stress.keras')
print('Model saved successfully')





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


