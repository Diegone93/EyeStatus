import sys
import os
# GRAPHIC CARD SETTINGS
if len(sys.argv) == 2:
    gpu_id = sys.argv[1]
else:
    gpu_id = "gpu0"
    cnmem = "0.3"
print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem
from batch_generators import load_names, load_images
from models import EyesStatusNet
import numpy as np
import cv2
import time
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # image parameters
    rows = 24
    cols = 24
    ch = 1

    # training parameters
    data_augmentation = False
    b_crop = False
    b_rescale = False
    b_scale = False
    b_normcv2 = True
    b_tanh = False

    batch_size = 2

    # image visualization
    b_visualize = True

    # graph visualization
    b_plot = True
    save = True

    # weights
    pesi = '..\weights\weights.018-0.13829.hdf5'

    # model
    model = EyesStatusNet(input_shape=(1, rows, cols),weights_path=pesi)

    # loading test sequence
    test_data_names = load_names(val_seq=-1)
    show = True
    contAccuracy = 0
    for image in range(len(test_data_names)):
        seq = test_data_names[image]['image'].split('\\')[-3]
        t = time.time()

        test_data_X, _ = load_images(test_data_names[image:image+1], crop=b_crop, rescale=b_rescale, scale=b_scale, b_debug=False,
                                     normcv2=b_normcv2, rows=rows, fulldepth=False, cols=cols, equalize=True,
                                     removeBackground=True,division=4)
        pred = model.predict(x=test_data_X, batch_size=batch_size, verbose=0)

        for index,i in enumerate(pred):

            gt_head = test_data_names[image+index]['eye']
            if gt_head[0] == 1:
                print('GT:      open')
                eyegt = True
            else:
                print('GT:      closed')
                eyegt = False
            if np.argmax(i) == 0:
                print('PREDICT: open')
                eye = True
            else:
                print('PREDICT: closed')
                eye = False
            if eye == eyegt:
                contAccuracy += 1

            print('_____________________')
            cv2.imshow('',cv2.resize(cv2.imread(test_data_names[image+index]['image'],cv2.IMREAD_GRAYSCALE),None,fx=10,fy=10))
            cv2.waitKey(1)
    print('final score: {}'.format(contAccuracy/float(len(test_data_names))))
