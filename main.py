import pandas as pd
import numpy as np
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import utils
from keras import regularizers
import matplotlib.pyplot as plt

# Reading and loading data
file1 = pd.read_csv('digit-recognizer/train.csv')
file2 = pd.read_csv('digit-recognizer/test.csv')
test_set = file2.values
test_x = test_set/255

training_set = np.random.permutation(file1.values)
Y = training_set[:, 0]
Y = utils.to_categorical(Y, num_classes = 10)

# Divide training set into train and validation sets 
val_y = Y[40000:42000, :]
train_Y = Y[0:40000, :]

X = training_set[:, 1:]/255
val_x = X[40000:42000, :]
train_X = X[0:40000, :]

train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
val_x = val_x.reshape(val_x.shape[0], 28, 28, 1)

# Layers
model = Sequential()
model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), data_format = 'channels_last', input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Dense(10, activation = 'softmax'))

# Callback for reducing learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.00002, verbose=1)
model_checkpoint = ModelCheckpoint(filepath = 'weights.hdf5', verbose = 1, save_best_only = True)

# Adam optimizer
optimizer = Adam(lr=0.0003)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics =['accuracy'])
history = model.fit(train_X, train_Y, epochs = 20, batch_size = 32, callbacks = [reduce_lr, model_checkpoint], verbose=2, validation_data = (val_x, val_y))

# Predictions
predictions = model.predict(test_x)

# Predict classes
predictions = np.argmax(predictions, 1)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Prepare the submission file
indices = np.arange(test_x.shape[0]) + 1
data = {'ImageId': indices, 'Label': predictions}
df = pd.DataFrame(data)
df.to_csv('digits.csv', index = False)
