from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

file_path_Train = "/content/dataset/training_set"
file_path_Test = "/content/dataset/test_set"

base = MobileNetV2(input_shape=(224, 224, 3),
                   include_top=False,
                   weights='imagenet')
base.trainable = False        # 1º estágio: congela convoluções
classificador = Sequential([
    base,
    GlobalAveragePooling2D(),  # substitui o Flatten gigantesco
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
classificador.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.10)      # ← 10% of training_set will be for validation
batch = 64


train_ds = datagen.flow_from_directory(
     file_path_Train,
     target_size=(224, 224),
     batch_size=batch,
     subset='training',
     class_mode='binary')

val_ds = datagen.flow_from_directory(
     file_path_Train,
     target_size=(224, 224),
     batch_size=batch,
     subset='validation',
     class_mode='binary')

classificador.summary()

steps_per_epoch = train_ds.samples // batch
val_steps       = val_ds.samples   // batch

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early   = EarlyStopping(patience=5, restore_best_weights=True)
plateau = ReduceLROnPlateau(patience=3, factor=0.2)
classificador.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=30,                       # EarlyStopping corta se não melhorar
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=[early, plateau])

#Tests:

file_path_Test = "/content/Teste.jpg"
img = image.load_img(file_path_Test, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

img #To show the image

proba = classificador.predict(img_array)[0][0]   # value ∈ [0,1]

if proba >= 0.9:
    print(f"Previsão: {proba:.4f}  →  it's a cat")
elif proba <=0.01:
    print(f"Previsão: {proba:.4f}  →  it's a dog")
else:
    print(f"Previsão: {proba:.4f}  →  probably has no dog or cat, or illegible")
