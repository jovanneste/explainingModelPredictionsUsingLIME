import os 
import numpy as np
import keras
from keras.applications import inception_v3 as inc_net
import keras.utils as image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt

# load model from keras
model = inc_net.InceptionV3()

img = image.load_img('elephant.jpg', target_size=(299,299))
out=[]
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inc_net.preprocess_input(x)
out.append(x)
images = np.vstack(out)

# how Inception represents images
plt.imshow(images[0] / 2 + 0.5)
plt.show()

predictions = inet_model.predict(images)
for x in decode_predictions(predictions)[0]:
    print(x)