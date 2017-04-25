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
print "Start computation in NLP machhine"

# Pick an image and show the image.
#sampleImageIndex = 5006 # Try changing this number and visualizing some other images from the dataset.
image_in_train = 10000
image_in_val = 100
# Compute features for the training set.
train_features = np.zeros((image_in_train, 50176), dtype=np.float)  # 768 = 16 * 16 * 3
#for (counter, image_id) in enumerate(train_data['images']):
for counter in range(0,image_in_train):
    image_id = train_data['images'][counter]
    image = imread('mscoco/%s' % image_id)
    image = color.rgb2gray(image)
    #tiny_image = imresize(image, (16, 16), interp = 'nearest')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    train_features[counter, :] = hog_image.flatten().astype(np.float)

    if (1 + counter) % 1000 == 0:
        print ('Computed features for %d train-images' % (1 + counter))

print "Compute features for the validation set."
val_features = np.zeros((image_in_val, 50176), dtype=np.float)  # 768 = 16 * 16 * 3
for counter in range(0,image_in_val):
    image_id = train_data['images'][counter]
    image = imread('mscoco/%s' % image_id)
    image = color.rgb2gray(image)
    #tiny_image = imresize(image, (16, 16), interp = 'nearest')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=True)
    val_features[counter, :] = hog_image.flatten().astype(np.float)
    if (1 + counter) % 100 == 0:
        print ('Computed features for %d val-images' % (1 + counter))

    #print len(hog_image)
    #print len(hog_image[0])
    #print counter
'''
print ("Start dumping train")
print len(train_data['images'])
sampleTestImageId = 2
# Retrieve the feature vector for this image.
sampleImageFeature = val_features[sampleTestImageId: sampleTestImageId + 1, :]
print len(sampleImageFeature)
print sampleImageFeature
pickle.dump(train_features, open('train_features_hog_gg.p', 'w'))  # Store in case this notebook crashes.
print ("Start dumping val")
pickle.dump(val_features, open('val_features_hog_gg.p', 'w'))  # Store in case this notebook crashes.
'''
'''
print ("Start dumping train")
pickle.dump(train_features, open('train_features_hog_gg.p', 'w'))  # Store in case this notebook crashes.
print ("Start dumping val")
pickle.dump(val_features, open('val_features_hog_gg.p', 'w'))  # Store in case this notebook crashes.
'''
#load tain and validation features
print 'Start Loading'
#val_features = pickle.load(open('val_features.p','rb'))
#train_features = pickle.load(open('train_features.p', 'rb'))
total_bleu_score = 0

print 'Start Retrieving'
total_images = image_in_val
nn_index = 0
start_time = time.time()
for index in range(0,total_images):
    sampleTestImageId = index
    # Retrieve the feature vector for this image.
    sampleImageFeature = val_features[sampleTestImageId: sampleTestImageId + 1, :]
    # Compute distances between this image and the training set of images.
    distances = cdist(sampleImageFeature, train_features, 'euclidean')
    # Compute ids for the closest images in this feature space.
    nearestNeighbors = np.argsort(distances[0, :])  # Retrieve the nearest neighbors for this image.
    #print ("NN = "+str(len(nearestNeighbors)))

    #nn_index = random.randint(0, 49999) #for random
    reference = [w.lower() for w in word_tokenize(val_data['captions'][sampleTestImageId])]
    candidate = [w.lower() for w in word_tokenize(train_data['captions'][nearestNeighbors[0]])]

    #print ('ref', reference)
    #print ('cand', candidate)

    bleu_score = float(len(set(reference) & set(candidate))) / len(candidate)
    total_bleu_score += bleu_score
    if index %10 == 0:
        print index
        print("Average BLEU-1 score HOG-Feature = ", (total_bleu_score * 1.0) / (index+1))
print("Average BLEU-1 score HOG-Feature = ", (total_bleu_score*1.0)/total_images)
print("Time taken: %s seconds " % (time.time() - start_time))