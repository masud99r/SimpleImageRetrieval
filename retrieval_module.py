import pickle
import numpy as np
from scipy.spatial.distance import cdist
from nltk import word_tokenize
# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']
print "Start computation for retrieval process"

#load tain and validation features
print 'Start Loading'
val_features = pickle.load(open('only_val_features_hog.p','rb'))
train_features = pickle.load(open('only_train_features_hog.p', 'rb'))
total_bleu_score = 0

print 'Start Retrieving'
total_images = 10000
nn_index = 0
for index in range(0,total_images):
    sampleTestImageId = index
    # Retrieve the feature vector for this image.
    sampleImageFeature = val_features[sampleTestImageId : sampleTestImageId + 1, :]
    # Compute distances between this image and the training set of images.
    distances = cdist(sampleImageFeature, train_features, 'euclidean')
    # Compute ids for the closest images in this feature space.
    nearestNeighbors = np.argsort(distances[0, :])  # Retrieve the nearest neighbors for this image.

    reference = [w.lower() for w in word_tokenize(val_data['captions'][sampleTestImageId])]
    candidate = [w.lower() for w in word_tokenize(train_data['captions'][nearestNeighbors[0]])]

    bleu_score = float(len(set(reference) & set(candidate))) / len(candidate)
    total_bleu_score += bleu_score
    if index %500 == 0:
        print index
        print("Average BLEU-1 score HOG-Feature = ", (total_bleu_score * 1.0) / (index+1))
print("Final Average BLEU-1 score HOG-Feature = ", (total_bleu_score*1.0)/total_images)

