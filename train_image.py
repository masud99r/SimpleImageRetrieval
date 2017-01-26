import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy.spatial.distance import cdist
from nltk import word_tokenize
import random
from skimage.feature import hog
from skimage import data, color, exposure
#%matplotlib inline #need to check here
import time
# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']
print "Start computation in training"

train_features = np.zeros((len(train_data['images']), 50176), dtype=np.float)  # 768 = 16 * 16 * 3
time_elapsed = 0
start_loop = time.time()
for (counter, image_id) in enumerate(train_data['images']):

    image = imread('mscoco/%s' % image_id)
    image = color.rgb2gray(image)
    #tiny_image = imresize(image, (16, 16), interp = 'nearest')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    train_features[counter, :] = hog_image.flatten().astype(np.float)

    if (1 + counter) % 1000 == 0:
        time_taken = time.time() - start_loop
        start_loop = time.time()
        print ('Computed features for %d train-images' % (1 + counter))
        time_elapsed += time_taken
        per_entry_time = (time_elapsed*1.0)/counter
        entry_remain = len(train_data['images']) - counter
        approx_time_remaining = entry_remain*per_entry_time
        print ('\t Time elaspsed(s) = ' + str(time_elapsed))
        print ('\t approx_time_remaining(s) = ' +str(approx_time_remaining))

print "Start dumping training"
pickle.dump(train_features, open('only_train_features_hog.p', 'w'))  # Store in case this notebook crashes.
print "End dumping training"