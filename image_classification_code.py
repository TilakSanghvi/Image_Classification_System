from keras.datasets import cifar10
import matplotlib.pyplot as plt

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

print(train_Y[31])
plt.imshow(train_X[31])

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

train_x=train_X.astype('float32')
test_X=test_X.astype('float32') 

train_X=train_X/255.0
test_X=test_X/255.0

train_Y = np_utils.to_categorical(train_Y)
test_Y = np_utils.to_categorical(test_Y)

num_classes = train_Y.shape[1]

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(5,5),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

model.summary()

model.fit(x = train_X, y = train_Y,
    validation_data = (test_X,test_Y),
    epochs = 15, batch_size = 64)

_, acc = model.evaluate(train_X, train_Y)

model.save("model.h5")
