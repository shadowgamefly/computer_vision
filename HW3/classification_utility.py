
# coding: utf-8

# In[60]:

import keras
import h5py
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


# In[51]:

img_width, img_height = 150, 150
batch_size = 16
model = load_model('model.h5')


# In[74]:

cat_path = 'cats_and_dogs_medium/test/cat/image00044.jpg'
dog_path = 'cats_and_dogs_medium/test/dog/image00003.jpg'
imgD = image.load_img(dog_path, target_size=(150, 150))
imgC = image.load_img(cat_path, target_size=(150, 150))
x = image.img_to_array(imgD)
y = image.img_to_array(imgC)
x = np.expand_dims(x, axis=0)
y = np.expand_dims(y, axis=0)
predX = model.predict(x)
predY = model.predict(y)
if predX[0][0]==1.0:
    captionX = "dog"
else :
    captionX = "cat"

if predY[0][0]==1.0:
    captionY = "dog"
else :
    captionY = "cat"


plt.title("this is a " + captionX)
plt.imshow(imgD)
plt.show()

plt.title("this is a " + captionY)
plt.imshow(imgC)
plt.show()
