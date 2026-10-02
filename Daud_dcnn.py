import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# dcnn setting + datatset load krna
dataset_path = "dataset"
img_size = (64, 64)
batch_size = 8
epochs = 30
num_classes = 3

# AUGMENTATION
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

#  DATA train cod
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# VALIDATION DATA
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("Classes found:", train_data.class_indices)







# Model ki hidden layers
def build_daud_model():
    model = models.Sequential([

        # Block 1 NN KI Conovolutional layer
        layers.Conv2D(32, (3,3), padding='same', input_shape=(64,64,3)),
        #Batch Norm ki layer (regulates the input from convolu layer to next layer to kepp values stable mean variance nikal k)
        layers.BatchNormalization(),
        # Relu deeactivate the negitive values of the nodes
        layers.Activation('relu'),
        #reduce image size by 2 x 2 pixel and keep the important features (eg security features in a bank note)
        layers.MaxPooling2D(2,2),
        # 25 % neuron ko deactivate krna so other neurons ko independent bnai
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.4),

        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes
    ])
    return model

model = build_daud_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("Daud's model ready!")









#  MODEL trainin
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

print("Training complete!")

# RESULTS
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Daud's Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Daud's Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('Daud_results.png')
plt.show()




model.save('Daud_model.keras')
print("Model saved as Daud_model.keras!")







#py -3.11 Daud_dcnn.py