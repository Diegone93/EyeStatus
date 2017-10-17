import numpy as np
import re
import cv2
import os
from sklearn import preprocessing
from skimage import exposure
import random

random.seed(1769)
DataValidationFolder =['open_test', 'closed_test']

DataTestFolder =['open_test', 'closed_test']

def load_names(val_seq = -1, augm=0):
    gt_dir = 'C:\Users\Diego\Desktop\eyeStatus\dataset_B_Eye_Images\\'

    if val_seq < 0:
        # load and remove validation sequence
        gt_list = os.listdir(gt_dir)
        deleting = DataValidationFolder
        for el in deleting:
            to_remove = os.path.join(el)
            gt_list.remove(to_remove)
    else:
        gt_list = DataTestFolder

    data = []

    for gt_folder in gt_list:
        for i, gt_file in enumerate(sorted(os.listdir(os.path.join(gt_dir, gt_folder)), key=lambda x: (int(re.sub('\D','',x)),x))):

            img_name = os.path.join(gt_dir,gt_folder,gt_file)
            if gt_folder == 'open' or gt_folder == 'open_test':
                data.append({'image': img_name, 'eye': [1,0], 'augm': augm})
            elif gt_folder == 'closed' or gt_folder == 'closed_test':
                data.append({'image': img_name, 'eye': [0, 1], 'augm': augm})

    random.shuffle(data)
    return data

def load_names_val(dataset=0):
    return load_names(val_seq=1)

def identity(img):
    return img

def Flip(img):
    return cv2.flip(img, 1)

def Traleft(img):
    M = np.float32([[1, 0, -(img.shape[0]/4)], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Traright(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Traup(img):
    M = np.float32([[1, 0, 0], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Tradown(img):
    M = np.float32([[1, 0, 0], [0, 1, (img.shape[0]/4 )]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Traleftup(img):
    M = np.float32([[1, 0, -(img.shape[0] / 4)], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Trarightup(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Traleftdown(img):
    M = np.float32([[1, 0, -(img.shape[0] / 4)], [0, 1, (img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Trarightdown(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, (img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def Rot45Left(img):
    image_center = tuple(np.array(img.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 45.0, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape, flags=cv2.INTER_LINEAR)

def Rot45Right(img):
    image_center = tuple(np.array(img.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -45.0, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape, flags=cv2.INTER_LINEAR)

# map the inputs to the function blocks
Augmentation = {
    0: identity,
    1 : Tradown,
    2 : Traup,
    3 : Traleft,
    4 : Traright,
    5 : Flip,
    6 : Traleftdown,
    7 : Traleftup,
    8 : Trarightdown,
    9 : Trarightup,
    10 : Rot45Left,
    11 : Rot45Right,
}

def load_images(train_data_names, crop, scale, rescale, normcv2, b_debug,fulldepth, rows, cols,equalize,removeBackground,division=8):

    # channel
    ch = 1

    # image structure
    img_batch = np.zeros(shape=(len(train_data_names), ch, rows, cols), dtype=np.float32)

    # GT structure
    y_batch = np.zeros(shape=(len(train_data_names),  2), dtype=np.float32)


    for i, line in enumerate(train_data_names):
        # image name
        img_name = line['image']
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        # Normalize (openCV)
        if normcv2:
            img = cv2.equalizeHist(img)
        # Rescale
        if rescale:
            img = exposure.rescale_intensity(img.astype('float'), in_range=(np.min(img), np.max(img)), out_range=(0, 1))

        # Scale
        if scale:
            img = preprocessing.scale(img.astype('float'))

        img = Augmentation[line['augm']](img)


        if b_debug:
            cv2.imshow("caricanda", (img))
            cv2.waitKey()

        # resize
        img = cv2.resize(img, (cols, rows))
        gt = line['eye']
        # add channel dimension
        img = np.expand_dims(img, 2)
        img = img.astype(np.float32)
        # batch loading
        img_batch[i] = img.transpose(2, 0, 1)
        y_batch[i] = gt

    return img_batch, y_batch
