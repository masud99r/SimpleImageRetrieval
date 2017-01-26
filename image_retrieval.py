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

# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']


# Pick an image and show the image.
#sampleImageIndex = 5006 # Try changing this number and visualizing some other images from the dataset.

# Compute features for the training set.
train_features = np.zeros((len(train_data['images']), 50176), dtype=np.float)  # 768 = 16 * 16 * 3
for (counter, image_id) in enumerate(train_data['images']):
    image = imread('mscoco/%s' % image_id)
    image = color.rgb2gray(image)
    #tiny_image = imresize(image, (16, 16), interp = 'nearest')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    #print len(hog_image)
    #print len(hog_image[0])
    train_features[counter, :] = hog_image.flatten().astype(np.float) / 255

    if (1 + counter) % 10000 == 0:
        print ('Computed features for %d train-images' % (1 + counter))

# Compute features for the validation set.
val_features = np.zeros((len(val_data['images']), 50176), dtype=np.float)  # 768 = 16 * 16 * 3
for (counter, image_id) in enumerate(val_data['images']):
    image = imread('mscoco/%s' % image_id)
    image = color.rgb2gray(image)
    #tiny_image = imresize(image, (16, 16), interp = 'nearest')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    val_features[counter, :] = hog_image.flatten().astype(np.float) / 255
    if (1 + counter) % 10000 == 0:
        print ('Computed features for %d val-images' % (1 + counter))

pickle.dump(train_features, open('train_features_hog.p', 'w'))  # Store in case this notebook crashes.
pickle.dump(val_features, open('val_features_hog.p', 'w'))  # Store in case this notebook crashes.


'''
# Try changing this image.
sampleTestImageId = 9999

#load tain and validation features
print 'Start Loading'
val_features = pickle.load(open('val_features.p','rb'))
train_features = pickle.load(open('train_features.p', 'rb'))
total_bleu_score = 0

print 'Start Retrieving'
total_images = 10000
nn_index = 0
for index in range(0,total_images):
    sampleTestImageId = index
    # Retrieve the feature vector for this image.
    sampleImageFeature = val_features[sampleTestImageId : sampleTestImageId + 1, :]
    # Compute distances between this image and the training set of images.
    distances = cdist(sampleImageFeature, train_features, 'correlation')
    # Compute ids for the closest images in this feature space.
    nearestNeighbors = np.argsort(distances[0, :])  # Retrieve the nearest neighbors for this image.
    #print ("NN = "+str(len(nearestNeighbors)))

    nn_index = random.randint(0, 49999)
    reference = [w.lower() for w in word_tokenize(val_data['captions'][sampleTestImageId])]
    candidate = [w.lower() for w in word_tokenize(train_data['captions'][nearestNeighbors[0]])]

    #print ('ref', reference)
    #print ('cand', candidate)

    bleu_score = float(len(set(reference) & set(candidate))) / len(candidate)
    total_bleu_score += bleu_score
    if index %500 == 0:
        print index
print("Average BLEU-1 score = ", (total_bleu_score*1.0)/total_images)
'''
