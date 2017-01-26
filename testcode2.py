from PIL import Image
import leargist

im = Image.open('mscoco/COCO_train2014_000000000025.jpg')
descriptors = leargist.color_gist(im)

print len(descriptors)
print descriptors.shape
print descriptors.dtype
print descriptors[:4]
