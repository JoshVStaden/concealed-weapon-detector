from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Resizing

import numpy as np

# image_gen = ImageDataGenerator(rescale=1/255.,rotation_range=0,width_shift_range=0.01,height_shift_range=0.01,
#                                horizontal_flip=True,vertical_flip=True,validation_split=0.2)
train_datagen = image_dataset_from_directory('../../Datasets/OD-WeaponDetection/pistol_detection',image_size=(224,224), label_mode='categorical',
                                              batch_size = 16, shuffle = True, validation_split=0.2, subset='training', seed=42)
val_datagen = image_dataset_from_directory('../../Datasets/OD-WeaponDetection/pistol_detection',image_size=(224,224), label_mode='categorical',
                                              batch_size = 16, shuffle = True, validation_split=0.2, subset='validation', seed=42)
print(dir(train_datagen))
print(train_datagen.class_names)
# train_datagen.classes[train_datagen.classes != 0] = 1
# print(train_datagen.classes)
# quit()
# val_datagen = image_gen.flow_from_directory('../../Datasets/OD-WeaponDetection/Pistol Classification',target_size=(224,224),class_mode='binary',
#                                               batch_size = 32, shuffle = True, subset = 'validation')
num_classes = len(np.unique(train_datagen.class_names))
pistol_class = [1] + [0] * (num_classes - 1)
new_classes = []

r_50 = VGG16(weights='imagenet', classes=num_classes, include_top=False, input_shape=(224,224,3))
r_50.trainable = False
model = Sequential()
model.add(Resizing(224, 224))
model.add(r_50)
# last_layer_shape = None
# for l in r_50.layers:
#     model.add(l)
#     last_layer_shape = l.shape
# model.add(Input((None, None, None, 512)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(train_datagen, validation_data=val_datagen, epochs=5)

model.save("pistol_classification")