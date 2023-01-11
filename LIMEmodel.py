import os 
import numpy as np
import keras
from keras.applications import inception_v3 as inc_net
import keras.utils as image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import lime 
from lime import lime_image

# load model from keras
model = inc_net.InceptionV3()

img = image.load_img('images/elephant.jpg', target_size=(299,299))
out=[]
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inc_net.preprocess_input(x)
out.append(x)
images = np.vstack(out)

# how Inception represents images
plt.imshow(images[0] / 2 + 0.5)
plt.show()

# predictions from pre-trained model
predictions = model.predict(images)
for x in decode_predictions(predictions)[0]:
    print(x)

# use LIME to explain the predictions

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

# overlay superpuxels for top most prediction on original image
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()