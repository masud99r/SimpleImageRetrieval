import pickle
import numpy as np
from scipy.spatial.distance import cdist
from nltk import word_tokenize
import random
from PIL import Image
import leargist

# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']

# Compute features for the training set.
train_features = np.zeros((len(train_data['images']), 960), dtype=np.float)  # 768 = 16 * 16 * 3
for (counter, image_id) in enumerate(train_data['images']):
    im = Image.open('mscoco/%s' % image_id)
    descriptors = leargist.color_gist(im)
    train_features[counter, :] = descriptors
    if (1 + counter) % 1000 == 0:
        print ('Computed features for %d train-images' % (1 + counter))

print "Compute features for the validation set."
val_features = np.zeros((len(val_data['images']), 960), dtype=np.float)  # 768 = 16 * 16 * 3
for (counter, image_id) in enumerate(val_data['images']):
    im = Image.open('mscoco/%s' % image_id)
    descriptors = leargist.color_gist(im)

    val_features[counter, :] = descriptors
    if (1 + counter) % 1000 == 0:
        print ('Computed features for %d val-images' % (1 + counter))

print ("Start dumping traind feature")
pickle.dump(train_features, open('train_features_hog_gg.p', 'w'))  # Store in case this notebook crashes.
print ("Start dumping validation feature")
pickle.dump(val_features, open('val_features_hog_gg.p', 'w'))  # Store in case this notebook crashes.

#Start retrieval
total_bleu_score = 0
print 'Start Retrieving'
total_images = 10000
for index in range(0,total_images):
    sampleTestImageId = index
    # Retrieve the feature vector for this image.
    sampleImageFeature = val_features[sampleTestImageId : sampleTestImageId + 1, :]
    # Compute distances between this image and the training set of images.
    distances = cdist(sampleImageFeature, train_features, 'correlation')
    # Compute ids for the closest images in this feature space.
    nearestNeighbors = np.argsort(distances[0, :])  # Retrieve the nearest neighbors for this image.
    reference = [w.lower() for w in word_tokenize(val_data['captions'][sampleTestImageId])]
    candidate = [w.lower() for w in word_tokenize(train_data['captions'][nearestNeighbors[0]])]

    #print ('ref', reference)
    #print ('cand', candidate)

    bleu_score = float(len(set(reference) & set(candidate))) / len(candidate)
    total_bleu_score += bleu_score
    if index %500 == 0:
        print index
print("Average BLEU-1 score gist= ", (total_bleu_score*1.0)/total_images)

