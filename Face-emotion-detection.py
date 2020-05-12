# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:59:57 2020

@author: Himanshu Panwar
"""


import sys, os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow 

BASEPATH = r'C:\Users\Himanshu\DeepEmotion-master\DeepEmotion-master\challenges-in-representation-learning-facial-expression-recognition-challenge'
sys.path.insert(0, BASEPATH)
os.chdir(BASEPATH)
MODELPATH = './models/model.h5'


# Initializing the parameters.
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48


data = pd.read_csv('./fer2013.csv')
data.head()


# 1. Converting the relevant column element into a list for each row
# 2. Splitting the string by space character as a list
# 3. Numpy ❤
# 4. Normalizing the image
# 5. Resizing the image
# 6. Expanding the dimension of channel for each image
# 7. Converting the labels to catergorical matrix
pixels = data['pixels'].tolist() # 1

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # 2
    face = np.asarray(face).reshape(width, height) # 3
    
    # There is an issue for normalizing images. Just comment out 4 and 5 lines until when I found the solution.
    # face = face / 255.0 # 4
    face = cv2.resize(face.astype('uint8'), (width, height)) # 5
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1) # 6

emotions = pd.get_dummies(data['emotion']).values # 7


# We are now ready to split our model into training, validation and test sets 
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

# import pillow as pil
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# import PIL
# from PIL import Image

# What about the architecture of the model will be?
# [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
# [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
# [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
# [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
# Dense (512) — DROPOUT (0.5)
# Dense (256) — DROPOUT (0.5)
# Dense (128) — DROPOUT (0.5)
# In the first convolutional layer, L2 regularization (0.01) has been added.
# In all convolutional layers except the first one, batch normalization layer has been added.
# MAXP (2x2) and DROPOUT (0.5) layers have been added to each convolutional layers block.
# “RELU” has been picked as activation function for all convolutional layers.
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.summary()

# compile model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])


# Let’s add some more features to our model.

# Firstly, we help the loss function to get rid of the “plateaus” by reducing the learning rate parameter of the optimization function with a certain value (factor) if there is no improvement on the value of the loss function for the validation set after a certain epoch (patience).
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

# We record everything done during the training into the “logs” folder as log to be able to better interpret the results of our model and to visually analyze the changes in the loss function and the accuracy during the training.
# For more information on TensorBoard: GO
tensorboard = TensorBoard(log_dir=BASEPATH)

# Even if we could prevent that the loss function goes to the plateaus, the value of the loss function of validation set could get stuck in a certain range while the training set’s does not (in other words, while the model continues to learn something). As long as we continue to train the model after this point, the only thing the model could do is to memorize (over-fit) the training data -I could say there is no chance of getting rid of the local minima for the loss function without a miracle-. This is something that we will not want at all.
# We stop the training of the model if there is no change in the value of the loss function on the validation set for a certain epoch (patience).
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

# Finally, we save our model during training as long as it gets a better result than the previous epoch. Thus, we will have the best possible model at the end of the training.
checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)


# We can start training our model. GO GO GO!!!
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_test), np.array(y_test)),
          shuffle=True,
          callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])

# model evaluation
scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

# The demo of predicting the facial expression of detected faces by Haar-Cascade face detection algorithm by using our trained model:
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model = load_model(MODELPATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
quit()


