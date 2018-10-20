from keras.models import load_model
import cv2
import numpy as np
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights.h5")
print("Loaded model from disk")


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('test.jpg')
img = cv2.resize(img,(320,240))
img = np.reshape(img,[1,320,240,3])

classes = model.predict_classes(img)

print classes
