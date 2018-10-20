'''Transfer learning: Phase 2 ..
    (Re-training the stripped VGG16)

    1) take the pretrained fully connected classifier (generated in classifier.py)
    2) connect this to the stripped VGG 16 CNN
    4) retrain the full network

Data directory structure (required) is shown below


```
data/
    train/
        benign/
            image1.jpg
            image2.jpg
            ...
        malignant/
            image1.jpg
            image2.jpg
            ...
    test/
        benign/
            image1.jpg
            image2.jpg
            ...
        malignant/
            image1.jpg
            image2.jpg
            ...

Edit the values for:

nb_train_samples
nb_validation_samples

below, to correctly reflect the number of images in the data directory


```
'''

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense


# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
full_model_weights_path = 'weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '../../ISIC-Dataset-Downloader-master/5a2ecc5d1165975c9459427e/'
validation_data_dir = '../../ISIC-Dataset-Downloader-master/5a2ecc5d1165975c9459427e/'
nb_train_samples = 2180
nb_validation_samples = 600
epochs = 50
batch_size = 16

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

model.summary()

# fine-tune the model
history = model.fit_generator(
		    train_generator,
		    steps_per_epoch=nb_train_samples // batch_size,
		    epochs=epochs,
		    validation_data=validation_generator,
		    validation_steps=nb_validation_samples // batch_size,
		    verbose=2)

model.save_weights(full_model_weights_path)

# list all data in history
print(history.history.keys())


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('AccuracyHistory.png')
#plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('LossHistory.png')
#plt.show()
