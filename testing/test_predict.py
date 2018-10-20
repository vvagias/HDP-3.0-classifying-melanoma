from keras.models import model_from_json
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import sys


# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
image_path = sys.argv[1]

orig = cv2.imread(image_path)

print("[INFO] loading and preprocessing image...")
image = load_img(image_path, target_size=(150, 150))
image = img_to_array(image)

# important! otherwise the predictions will be '0'
image = image / 255

image = np.expand_dims(image, axis=0)

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16


datagen = ImageDataGenerator(rescale=1. / 255)

generator = datagen.flow_from_directory(
        '/Volumes/Seagate Backup Plus Drive/DevTools/code_base/melanoma/HDP-3.0-classifying-melanoma/testing/test_images',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

datagen_top = ImageDataGenerator(rescale=1./255)

generator_top = datagen_top.flow_from_directory(
     '/Volumes/Seagate Backup Plus Drive/DevTools/code_base/melanoma/HDP-3.0-classifying-melanoma/testing/test_images',
     target_size=(img_width, img_height),
     batch_size=batch_size,
     class_mode='categorical',
     shuffle=False)
 # build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

 # get the bottleneck prediction from the pre-trained VGG16 model
bottleneck_prediction = model.predict(image)

 # build top model
model = Sequential()
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.load_weights(top_model_weights_path)


# use the bottleneck prediction on the top model to get the final classification
class_predicted = model.predict_classes(bottleneck_prediction)

inID = class_predicted[0]
print(generator_top.class_indices)
class_dictionary = generator_top.class_indices

inv_map = {v: k for k, v in class_dictionary.items()}

label = inv_map[inID[0]]

# get the prediction label
print("Image ID: {}, Label: {}".format(inID, label))

# display the predictions with the image
cv2.putText(orig, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

cv2.imshow("Classification", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
