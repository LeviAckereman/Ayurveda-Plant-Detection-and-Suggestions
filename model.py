import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from math import ceil
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Define Dataset Directory
main_data_dir = "D:/ADITYA/BE PROJECT/Ayurveda Website/dataset"
batch_size = 32
epochs = 30

# Count number of classes correctly
num_classes = len([d for d in os.listdir(main_data_dir) if os.path.isdir(os.path.join(main_data_dir, d))])

# Define ImageDataGenerator with VGG19 preprocessing and more advanced augmentations
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # VGG19-specific preprocessing
    rotation_range=20,   # Random rotations from -40 to 40 degrees
    width_shift_range=0.2,  # Horizontal shift (0 to 20% of the image width)
    height_shift_range=0.2,  # Vertical shift (0 to 20% of the image height)
    shear_range=0.2,    # Shear transformation range (0 to 20%)
    zoom_range=0.3,     # Zoom in and out within a 30% range
    horizontal_flip=True,  # Randomly flip images horizontally
    brightness_range=[0.5, 1.5],  # Randomly change brightness between 50% and 150%
    fill_mode='nearest',  # Fill newly created pixels after transformations
    validation_split=0.2  # 20% for validation data
)

# Training and validation data loaders
train_generator = train_datagen.flow_from_directory(
    main_data_dir,
    target_size=(224, 224),  # Resize images to 224x224 for VGG19 input
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    main_data_dir,
    target_size=(224, 224),  # Resize images to 224x224 for VGG19 input
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Load VGG19 Model 
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze only the first 5 layers, fine-tune deeper layers
for layer in base_model.layers[:5]:
    layer.trainable = False
for layer in base_model.layers[5:]:
    layer.trainable = True

# Add Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
from tensorflow.keras.layers import BatchNormalization
x = BatchNormalization()(x)  # Add batch normalization
from tensorflow.keras import regularizers
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define Model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model 
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Adam optimizer with learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Learning Rate Scheduler for more gradual decay
def scheduler(epoch, lr):
    if epoch < 5:
        return float(lr)  # Keep initial learning rate for the first few epochs
    else:
        new_lr = lr * tf.math.exp(-0.1)  # Gradual decay after epoch 5
        return float(new_lr)  # Ensure returning a float value

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Add EarlyStopping to prevent overtraining
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    steps_per_epoch=ceil(train_generator.samples / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=ceil(validation_generator.samples / batch_size),
    callbacks=[early_stopping, lr_scheduler]
)

# Save Model
model.save('D:/ADITYA/BE PROJECT/Ayurveda Website/plant_model.h5')

# Evaluate Model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy:.2f}")

# Load Trained Model
model = tf.keras.models.load_model('D:/ADITYA/BE PROJECT/Ayurveda Website/plant_model.h5')

# Recompile Model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preprocessing Function for Image Prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Prediction Function
def predict_plant(image_path, label_mapping):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_label_index = int(np.argmax(predictions[0]))
    predicted_label = label_mapping[predicted_label_index]
    confidence = float(predictions[0][predicted_label_index])
    return predicted_label, confidence

# Label mapping for prediction
label_mapping = {0: 'Aloevera', 1: 'Amla', 2: 'Amrutaballi', 3: 'Arali', 4: 'Ashoka', 5: 'Ashwagandha',
                 6: 'Asthma weed', 7: 'Avocado', 8: 'Badipala', 9: 'Balloon Vine', 10: 'Bamboo', 11: 'Basale',
                 12: 'Beans', 13: 'Betel', 14: 'Bhringaraj', 15: 'Brahmi', 16: 'Camphor', 17: 'Caricature',
                 18: 'Castor', 19: 'Chakte', 20: 'Chilly', 21: 'Citron lime', 22: 'Coffee', 23: 'Common rue',
                 24: 'Coriander', 25: 'Curry', 26: 'Doddapatre', 27: 'Drumstick', 28: 'Ekka', 29: 'Eucalyptus',
                 30: 'Ganigale', 31: 'Ganike', 32: 'Gasagase', 33: 'Geranium', 34: 'Ginger', 35: 'Globe Amarnath',
                 36: 'Guava', 37: 'Henna', 38: 'Hibiscus', 39: 'Honge', 40: 'Insulin', 41: 'Jackfruit',
                 42: 'Jasmine', 43: 'Kamakasturi', 44: 'Kambajala', 45: 'Kasambruga', 46: 'Kepala', 47: 'Kohlrabi',
                 48: 'Lantana', 49: 'Lemon', 50: 'Lemongrass', 51: 'Malabar Nut', 52: 'Malabar Spinach',
                 53: 'Mango', 54: 'Marigold', 55: 'Mint', 56: 'Nagadali', 57: 'Neem', 58: 'Nelavembu',
                 59: 'Nerale', 60: 'Nithyapushpa', 61: 'Nooni', 62: 'Onion', 63: 'Padri', 64: 'Papaya',
                 65: 'Parijatha', 66: 'Pea', 67: 'Pepper', 68: 'Pomegranate', 69: 'Pumpkin', 70: 'Radish',
                 71: 'Raktachandini', 72: 'Rose', 73: 'Sampige', 74: 'Sapota', 75: 'Seethapala', 76: 'Spinach',
                 77: 'Tamarind', 78: 'Taro', 79: 'Tecoma', 80: 'Thumbe', 81: 'Tomato', 82: 'Tulsi', 83: 'Turmeric',
                 84: 'Wood Sorrel'}

# Example prediction
image_path = "D:\ADITYA\BE PROJECT\dataset\Ashoka\844.jpg" # Replace with actual image path
predicted_label, confidence = predict_plant(image_path, label_mapping)

# Print the prediction
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

# Plot training and validation loss and accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
val_preds = model.predict(validation_generator)
val_preds_labels = np.argmax(val_preds, axis=1)
print(classification_report(validation_generator.classes, val_preds_labels))

cm = confusion_matrix(validation_generator.classes, val_preds_labels)
print(cm)
