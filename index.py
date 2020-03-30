# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np

from extract_cnn_vgg16_keras import VGGNet


def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


if __name__ == "__main__":

    # db = args["database"]
    db = "database"
    img_list = get_imlist(db)
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    # output = args["index"]
    output = "featureCNN.h5"
    # print(feats)
    # print(feats.shape)
    print("--------------------------------------------------")
    # print(names)
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...      ")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()




