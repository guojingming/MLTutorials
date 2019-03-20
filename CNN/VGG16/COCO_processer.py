from pycocotools.coco import COCO

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/home/jlurobot/dl_ws/coco'
dataType = 'train2014'
annFile = '{0}/annotations/instances_{1}.json'.format(dataDir, dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
#imgIds = coco.getImgIds(imgIds=[3])


for imgId in imgIds:
    imgArray = coco.loadImgs(imgId);
    imgDict = imgArray[0]
    I = io.imread("{0}/{1}/{2}".format(dataDir, dataType, imgDict['file_name']))
    plt.axis('off')
    plt.imshow(I)
    plt.show()
    plt.pause(1)