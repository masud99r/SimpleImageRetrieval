import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy.spatial.distance import cdist
#%matplotlib inline #need to check here

# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']

'''
# Pick an image and show the image.
sampleImageIndex = 2000 # Try changing this number and visualizing some other images from the dataset.
plt.figure()
plt.imshow(imread('mscoco/%s' % train_data['images'][sampleImageIndex]))

print(sampleImageIndex, train_data['captions'][sampleImageIndex])
plt.show()
'''
