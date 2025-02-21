import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RANDOM_SEED = 42
dataset = 'gestures.csv'  # Path to your dataset
model_save_path_checkpoint = 'gesture_model.keras'
model_save_path_final = 'gesture_model.h5'
tflite_save_path = 'gesture_model.tflite'
NUM_CLASSES = 7  # Update based on the number of gestures you have collected

# Read the dataset
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=range(1, 43))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=0)

# Split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_dataset, y_dataset, test_size=0.3, random_state=RANDOM_SEED
)  # 70% train, 30% temp (validation + test)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
)  # Split temp into 50% validation, 50% test

# Print dataset sizes for verification
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Build the model
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42, )),  # 21 landmarks * 2 (x and y)
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path_checkpoint, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[cp_callback, es_callback]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

# Save the model
model.save(model_save_path_final, include_optimizer=False)
model.save("gesture_model.keras")

# Convert the model to TensorFlow Lite format (optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)
