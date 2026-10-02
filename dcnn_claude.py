import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# LOAD DATASET
# CIFAR-10 has 60,000 images in 10 categories like:
# airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training images shape:", X_train.shape)
print("Test images shape:", X_test.shape)







# NORMALIZE
# Pixels are 0-255 (like brightness levels)
# We divide by 255 to make them 0-1
# Small numbers are easier for the network to learn from
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# ONE-HOT ENCODE LABELS
# Converts: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# Like a C++ bool array where only one position is true
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print("Normalization done!")
print("Sample label:", y_train[0])












# BUILD THE DCNN MODEL
def build_model():
    model = models.Sequential()

    # --- BLOCK 1 ---
    # Conv2D: 32 filters scanning the image for basic features like edges
    model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())  # stabilizes learning
    model.add(layers.Activation('relu'))   # removes negatives
    model.add(layers.MaxPooling2D(2,2))    # shrinks image from 32x32 to 16x16
    model.add(layers.Dropout(0.25))        # randomly turns off 25% neurons to prevent overfitting

    # --- BLOCK 2 ---
    # 64 filters now - learning more complex shapes
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2,2))    # shrinks from 16x16 to 8x8
    model.add(layers.Dropout(0.25))

    # --- BLOCK 3 ---
    # 128 filters - learning high level features like "cat face"
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2,2))    # shrinks from 8x8 to 4x4
    model.add(layers.Dropout(0.4))

    # --- CLASSIFIER ---
    model.add(layers.Flatten())            # converts 4x4x128 → single long array
    model.add(layers.Dense(256, activation='relu'))  # fully connected layer
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax')) # 10 outputs = 10 categories

    return model

model = build_model()
model.summary()  # prints the full architecture









# COMPILE THE MODEL
# optimizer = how the model updates its weights (Adam is the best general choice)
# loss = how we measure how wrong the model is
# metrics = what we want to track (accuracy)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled and ready to train!")












# TRAIN THE MODEL
# epoch = one full pass through all 50,000 training images
# batch_size = how many images to process at once (64 at a time)
# validation_data = checks accuracy on test data after each epoch

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

print("Training complete!")











# EVALUATE ON TEST DATA
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# PLOT ACCURACY GRAPH
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
print("Graph saved as training_results.png!")



#py -3.11 dcnn_claude.py