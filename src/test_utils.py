import random
import imageio
import scipy.misc
import numpy as np
import IPython
import cv2

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.

def load_vimeo_data(data_path, image_size, K, T):
    seq = np.zeros((image_size, image_size, K+T, 3), dtype="float32")
    for t in range(0, K+T):
        img = cv2.imread(data_path + "/im%d.png"%(t + 1))
        # Below is the part for adapt the shape of image to right shape, right now I just take a simple indice
        # Here our input's cahnnel is 3, for YUV each
        yuv_img = cv2.cvtColor(img[:image_size, :image_size, :], cv2.COLOR_RGB2YUV)
        seq[:,:,t] = transform(yuv_img)
    IPython.embed()
    # Then cal the diff, though we only have one diff
    diff = np.zeros((image_size, image_size, K-1, 3), dtype="float32")
    for t in range(1, K):
        prev = inverse_transform(seq[:,:,t-1])
        cur  = inverse_transform(seq[:,:,t  ])
        diff[:,:,t - 1] = cur.astype("float32") - prev.astype("float32")
    return seq, diff


load_vimeo_data("/home/struct/iku/iclr2017mcnet/data/vimeo_interp_test/target/00001/0389", 200, 2, 1)