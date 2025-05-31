import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from keras import callbacks
import keras_tuner as kt
from keras_tuner import HyperModel 
import os
from sklearn.utils import class_weight, resample
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import precision_score, recall_score

# Data loading
csv_path_train = os.path.join('Database', 'trainingdata_basic.csv')
csv_path_test = os.path.join('Database', 'testdata_basic.csv')

df_train = pd.read_csv(csv_path_train)
df_test = pd.read_csv(csv_path_test)

# Fix image paths
df_train['Images'] = df_train['Images'].apply(lambda x: os.path.join('Database', x.split('Database/')[-1]))
df_test['Images'] = df_test['Images'].apply(lambda x: os.path.join('Database', x.split('Database/')[-1]))

print("Original class distribution:")
print(df_train['Label'].value_counts().sort_index())
print(f"Imbalance ratio: {df_train['Label'].value_counts().max() / df_train['Label'].value_counts().min():.1f}:1")

# Strategy 1: Balanced sampling with data augmentation for minority classes
def balance_dataset_with_augmentation(df, target_samples_per_class=1500):
    """Balance dataset by oversampling minority classes with heavy augmentation"""
    balanced_dfs = []
    
    for label in sorted(df['Label'].unique()):
        class_df = df[df['Label'] == label].copy()
        current_count = len(class_df)
        
        if current_count < target_samples_per_class:
            # Calculate how many times to oversample
            oversample_factor = target_samples_per_class // current_count
            remainder = target_samples_per_class % current_count
            
            # Repeat the entire class multiple times
            oversampled_parts = [class_df] * oversample_factor
            
            # Add random remainder samples
            if remainder > 0:
                remainder_samples = class_df.sample(n=remainder, replace=True, random_state=42)
                oversampled_parts.append(remainder_samples)
            
            # Combine all parts
            balanced_class = pd.concat(oversampled_parts, ignore_index=True)
            print(f"Class {label}: {current_count} -> {len(balanced_class)} samples")
        else:
            # Downsample majority classes slightly
            balanced_class = class_df.sample(n=target_samples_per_class, random_state=42)
            print(f"Class {label}: {current_count} -> {len(balanced_class)} samples (downsampled)")
        
        balanced_dfs.append(balanced_class)
    
    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Balance the training data
balanced_train_df = balance_dataset_with_augmentation(df_train, target_samples_per_class=1500)

# Create validation split from balanced data
train_df, val_df = train_test_split(balanced_train_df, test_size=0.15, stratify=balanced_train_df['Label'], random_state=42)

print(f"\nAfter balancing:")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(df_test)}")
print("\nBalanced training class distribution:")
print(train_df['Label'].value_counts().sort_index())

# Enhanced data preprocessing with class-specific augmentation
def load_and_preprocess_image(image_path, label, is_training=True):
    # Read and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    
    # Apply stronger augmentation for minority classes during training
    if is_training:
        # Get class counts for adaptive augmentation
        minority_classes = [2, 3, 6]  # Fear, Disgust, Anger (lowest counts)
        
        # Standard augmentation for all
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        
        # Additional augmentation for minority classes
        if tf.reduce_any(tf.equal(label, minority_classes)):
            # More aggressive augmentation for rare classes
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.1)
            # Random rotation
            angle = tf.random.uniform([], -0.2, 0.2)
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    
    # Normalize to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Convert label (1-indexed to 0-indexed)
    label = tf.cast(label - 1, tf.int32)
    
    return image, label

# Build datasets
def create_dataset(df, is_training=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((df['Images'].values, df['Label'].values))
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(2000, len(df)))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create datasets with smaller batch size for better gradient updates
batch_size = 32
train_dataset = create_dataset(train_df, is_training=True, batch_size=batch_size)
val_dataset = create_dataset(val_df, is_training=False, batch_size=batch_size)
test_dataset = create_dataset(df_test, is_training=False, batch_size=batch_size)

# Compute class weights for the original imbalanced data (for test evaluation)
original_labels = np.unique(df_train['Label'])
original_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=original_labels,
    y=df_train['Label']
)
class_weights_dict = {label - 1: weight for label, weight in zip(original_labels, original_class_weights)}
print(f"\nClass weights (for original imbalance): {class_weights_dict}")

# Custom Focal Loss for handling remaining imbalance
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Convert sparse labels to one-hot
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[1])
        
        # Compute focal loss
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
        p_t = tf.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss

# Simplified approach - use basic Keras metrics and calculate macro metrics separately
def create_per_class_metrics(num_classes=7):
    """Create individual precision and recall metrics for each class"""
    metrics = ['accuracy']
    
    # Add per-class precision metrics
    for i in range(num_classes):
        metrics.append(tf.keras.metrics.Precision(class_id=i, name=f'precision_class_{i}'))
        metrics.append(tf.keras.metrics.Recall(class_id=i, name=f'recall_class_{i}'))
    
    return metrics

# Build improved model with attention mechanism
def build_imbalanced_model(num_classes=7):
    base_model = tf.keras.applications.ResNet101V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Preprocessing
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Add attention mechanism to focus on important features
    # Global Average Pooling
    gap = layers.GlobalAveragePooling2D()(x)
    
    # Attention weights
    attention = layers.Dense(x.shape[-1], activation='sigmoid')(gap)
    attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
    
    # Apply attention
    x_attended = layers.Multiply()([x, attention])
    
    # Pool the attended features
    x = layers.GlobalAveragePooling2D()(x_attended)
    
    # Classification head with batch normalization
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

# Build model
model, base_model = build_imbalanced_model()

# Use Focal Loss to handle remaining class imbalance
focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

# Compile with per-class metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss,
    metrics=create_per_class_metrics()
)

print("\nModel architecture:")
model.summary()

# Enhanced callbacks
callbacks_list = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_balanced_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Stage 1: Train with frozen base model
print("\n" + "="*60)
print("STAGE 1: Training with frozen base model (Balanced Data)")
print("="*60)

history_1 = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluate after stage 1
print("\nEvaluating after Stage 1...")
val_results_1 = model.evaluate(val_dataset, verbose=0)
print(f"Stage 1 - Val Loss: {val_results_1[0]:.4f}, Accuracy: {val_results_1[1]:.4f}")

# Calculate macro precision and recall manually
print("Calculating macro metrics...")
y_pred_val = []
y_true_val = []
for batch_images, batch_labels in val_dataset:
    predictions = model.predict(batch_images, verbose=0)
    y_pred_val.extend(np.argmax(predictions, axis=1))
    y_true_val.extend(batch_labels.numpy())

val_macro_precision = precision_score(y_true_val, y_pred_val, average='macro', zero_division=0)
val_macro_recall = recall_score(y_true_val, y_pred_val, average='macro', zero_division=0)
print(f"Stage 1 - Macro Precision: {val_macro_precision:.4f}, Macro Recall: {val_macro_recall:.4f}")

# Stage 2: Fine-tuning
print("\n" + "="*60)
print("STAGE 2: Fine-tuning with unfrozen layers")
print("="*60)

base_model.trainable = True

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss,
    metrics=create_per_class_metrics()
)

fine_tune_callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_finetuned_balanced_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history_2 = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=fine_tune_callbacks,
    verbose=1
)

# Final evaluation
print("\n" + "="*60)
print("FINAL EVALUATION ON ORIGINAL IMBALANCED TEST SET")
print("="*60)

test_results = model.evaluate(test_dataset, verbose=1)
test_loss = test_results[0]
test_acc = test_results[1]

print("Calculating test macro metrics...")
y_pred_test = []
y_true_test = []
for batch_images, batch_labels in test_dataset:
    predictions = model.predict(batch_images, verbose=0)
    y_pred_test.extend(np.argmax(predictions, axis=1))
    y_true_test.extend(batch_labels.numpy())

test_precision = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
test_recall = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)

print(f"\nFinal Test Results:")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test precision (macro): {test_precision:.4f}")
print(f"Test recall (macro): {test_recall:.4f}")
print(f"Test loss: {test_loss:.4f}")

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

# best_model.save('best_model_stress.keras')
# print('Model saved successfully')





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


