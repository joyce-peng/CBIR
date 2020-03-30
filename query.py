# -*- coding: utf-8 -*-
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import matplotlib.image as mpimg


args = {}
args.update(index="featureCNN.h5")
args.update(query="database/116400.jpg")
args.update(result="database")

h5f = h5py.File(args["index"],'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
# print(feats)
imgNames = h5f['dataset_2'][:]
print(imgNames)
h5f.close()

def get_groundtruth():
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    gt = []
    for line in open("holidays_images.dat", "r"):
        imname = line.strip()
        imno = int(imname[:-len(".jpg")])
        if imno % 100 == 0:
            gt.append(imname)
    return (gt)

gt = get_groundtruth()
print(gt)

# init VGGNet16 model
model = VGGNet()

for queryDir in gt:
    queryImg = mpimg.imread(str('database/'+ queryDir))
    queryVec = model.extract_feat(str('database/'+ queryDir))
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    maxres = 10
    print('-----------------------------')
    imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]

    f = open('peng_result.txt', "a")
    f.write(queryDir + ' ')
    for i, index in enumerate(rank_ID[0:maxres]):
        img = str(imgNames[index]).strip('b')
        img = img.strip('\'')
        f.write(' ' + str(i) + ' ' + str(img))
    f.write('\n')

    f.close()

    print("top %d images in order are: " % maxres, imlist)
    print(rank_ID[0:maxres])

