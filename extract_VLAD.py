import cv2
import os
import numpy as np
import heapq

####hyperparameter
COLUMNOFCODEBOOK = 32
DESDIM = 128
SUBVEC = 32
SUBCLUSTER = 256
PCAD = 128
TESTTYPE = 0

def get_allpath(FILE_PATH = "E:/pyWorkspace/VLAD/data/"):
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    file_path_list = []
    for line in open("holidays_images.dat", "r"):
        imname = line.strip()
        file_path_list.append(FILE_PATH + imname)
    return (file_path_list)


def get_groundtruth(FILE_PATH = "E:/pyWorkspace/VLAD/data/"):
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    file_path_list = []
    for line in open("holidays_images.dat", "r"):
        imname = line.strip()
        imno = int(imname[:-len(".jpg")])
        if imno % 100 == 0:
            file_path_list.append(FILE_PATH + imname)
    return (file_path_list)

def get_des_vector(file_path_list):
    '''
    Description: get descriptors of all the images
    Input: file_path_list - all images path
    Output:       all_des - a np array of all descriptors
            iamge_des_len - a list of number of the keypoints for each image
    '''
    all_des = np.empty(shape=[0, 128])
    image_des_len = []
    for eachFile in file_path_list:
            des = sift_extractor(eachFile)
            all_des = np.concatenate([all_des, des])
            image_des_len.append(len(des))
            # print(eachFile)
            # print(all_des)
            # print('-------------------------------')
    return all_des, image_des_len

def sift_extractor(file_path):
    '''
    Description: extract \emph{sift} feature from given image
    Input: file_path - image path
    Output: des - a list of descriptors of all the keypoint from the image
    '''
    img = cv2.imread(file_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    _,des = sift.detectAndCompute(gray,None)
    return des

def feature_extractor(file_path):
    '''
    Description: extract feature from given image
    Input: file_path - image path
    Output: des - a list of descriptors of all the keypoint from the image
    '''
    detector = cv2.ORB_create(nfeatures=500)
    img = cv2.imread(file_path)
    kp = detector.detect(img, None) #find the keypoint
    _, des = detector.compute(img, kp) #compute the descriptor
    return des

def get_codebook(all_des, K):
    '''
    Description: train the codebook from all of the descriptors
    Input: all_des - training data for the codebook
                 K - the column of the codebook
    '''
    label, center = get_cluster_center(all_des, K)
    return label, center

def get_cluster_center(des_set, K):
    '''
    Description: cluter using a default setting
    Input: des_set - cluster data
                 K - the number of cluster center
    Output: laber  - a np array of the nearest center for each cluster data
            center - a np array of the K cluster center
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    des_set = np.float32(des_set)
    ret, label, center = cv2.kmeans(des_set, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    return label, center

def get_vlad_base(img_des_len, NNlabel, all_des, codebook):
    '''
    Description: get all images vlad vector
    '''
    cursor = 0
    vlad_base = []
    for eachImage in img_des_len:
        descrips = all_des[cursor: cursor + eachImage]
        centriods_id = NNlabel[cursor: cursor + eachImage]
        centriods = codebook[centriods_id]
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM]) #(32,128)
        # print(eachImage)
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes]
            # print(eachDes)
            # print(vlad)
            # print('---------------------')
        cursor += eachImage

        vlad_norm = vlad.copy()
        cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
        vlad_base.append(vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1))

    return vlad_base

def get_pic_vlad(pic, des_size, codebook):
    '''
    Description: get the vlad vector of each image
    '''
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = 1000000000.0
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i])
            if dist < min_dist:
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind]

    vlad_norm = vlad.copy()
    cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
    vlad_norm = vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)

    return vlad_norm

def cal_vec_dist(vec1, vec2):
    '''
    Description: calculate the Euclidean Distance of two vectors
    '''
    return np.linalg.norm(vec1 - vec2)


##get all the descriptor vectors of the data set
gt = get_allpath()
file_path_list = gt
all_des, image_des_len = get_des_vector(gt)
# print(all_des.shape)
# print(image_des_len)

##get all the descriptor vectors of the query set
retrieval_image_path = []
retrieval_image_path = get_groundtruth()
# retrieval_image_path.append("E:/pyWorkspace/VLAD/query/103900.jpg")
ret_des, ret_des_len = get_des_vector(retrieval_image_path)
# print(ret_des.shape)
# print(ret_des_len)


##trainning the codebook

# NNlabel, codebook = get_codebook(all_des, COLUMNOFCODEBOOK)

##loading the codebook
fname_a = 'codebook.npy'
fname_b = 'NNlabel.npy'

codebook = np.load(fname_a)
NNlabel = np.load(fname_b)


# get_vlad_base(image_des_len, NNlabel, all_des, codebook)
vlad_base = get_vlad_base(image_des_len, NNlabel, all_des, codebook)
# print(vlad_base)
#get all the vlad vectors of retrival set without pca dimensionality reduction
cursor_ret = 0
ret_vlad_list = []
for eachretpic in range(len(ret_des_len)):
    pic = ret_des[cursor_ret: cursor_ret + ret_des_len[eachretpic]]
    ret_vlad = get_pic_vlad(pic, ret_des_len[eachretpic], codebook)
    cursor_ret += ret_des_len[eachretpic]
    ret_vlad_list.append(ret_vlad)

#test and evaluation
top1_cnt = 0
# print(len(ret_vlad_list))
# print(len(image_des_len))
for i in range(len(ret_vlad_list)):
    dist_list = []
    print("%dth image" % (i))
    for eachpic in range(len(image_des_len)):
        dist = cal_vec_dist(ret_vlad_list[i], vlad_base[eachpic])
        dist_list.append(dist)

        most_sim = np.array(dist_list)

        # choose the three nearest images of the given image
        z = heapq.nsmallest(10, most_sim)
        index_first = dist_list.index(z[0])
        index_second = dist_list.index(z[1])
        index_third = dist_list.index(z[2])
        index_fourth = dist_list.index(z[3])
        index_fifth = dist_list.index(z[4])
        index_sixth = dist_list.index(z[5])
        index_seventh = dist_list.index(z[6])
        index_eighth = dist_list.index(z[7])
        index_ninth = dist_list.index(z[8])
        index_tenth = dist_list.index(z[9])

        top1 = file_path_list[index_second][22:26]
        if top1 == str(i + 1000):
            top1_cnt += 1

        print("the %s is the first sim,the distance is the %f" % (file_path_list[index_first], z[0]))
        # print("the %s is the second sim,the distance is the %f" % (file_path_list[index_second], z[1]))
        # print("the %s is the second sim,the distance is the %f" % (file_path_list[index_third], z[2]))

        img1 = file_path_list[index_first].split("/")[-1]
        img2 = file_path_list[index_second].split("/")[-1]
        img3 = file_path_list[index_third].split("/")[-1]
        img4 = file_path_list[index_fourth].split("/")[-1]
        img5 = file_path_list[index_fifth].split("/")[-1]
        img6 = file_path_list[index_sixth].split("/")[-1]
        img7 = file_path_list[index_seventh].split("/")[-1]
        img8 = file_path_list[index_eighth].split("/")[-1]
        img9 = file_path_list[index_ninth].split("/")[-1]
        img10 = file_path_list[index_tenth].split("/")[-1]


        f = open('peng_result_vlad3.txt', "a")
        f.write(str(img1) + '  ')
        f.write('0' + ' ' + str(img1) + ' '
                    + '1' + ' ' + str(img2) + ' '
                    + '2' + ' ' + str(img3) + ' '
                    + '3' + ' ' + str(img4) + ' '
                    + '4' + ' ' + str(img5) + ' '
                    + '5' + ' ' + str(img6) + ' '
                    + '6' + ' ' + str(img7) + ' '
                    + '7' + ' ' + str(img8) + ' '
                    + '8' + ' ' + str(img9) + ' '
                    + '9' + ' ' + str(img10) + '\n')
        f.close()

    print(top1_cnt / 500.0)