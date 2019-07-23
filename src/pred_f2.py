import os
import cv2
import sys
import time
# import ssim
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(first, second, out):
    K=2
    T=1
    ckpt= "MCNET.model-40002"
    ref1 = first
    ref2 = second
    # IPython.embed()

    img1 = cv2.imread(ref1)
    img2 = cv2.imread(ref2)

    img1_yuv = cv2.cvtColor(img1, cv2.COLOR_RGB2YUV)
    img2_yuv = cv2.cvtColor(img2, cv2.COLOR_RGB2YUV)

    # Some basic setting
    height = img1_yuv.shape[0]
    width  = img1_yuv.shape[1]
    c_dim  = 3
    # Then begin to build the MCNet
    with tf.device("/cpu:0"):
        model = MCNET(image_size=[height, width], batch_size=1, K=K, T=T, c_dim=c_dim, is_train=False, checkpoint_dir=ckpt)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model.load_ckpt(sess, ckpt)
        
        # exit(0)
        seq = np.zeros([1, height, width, K+T, 3], dtype="float32")
        seq[...,0] = transform(img1_yuv)
        seq[...,1] = transform(img2_yuv)

        diff = np.zeros([1, height, width, K-1, 3], dtype="float32")
        diff[:,:,:,0,:] = inverse_transform(seq[...,1]) - inverse_transform(seq[...,0])

        pred_raw = sess.run([model.G], 
                            feed_dict={model.diff_in: diff,
                                        model.xt: seq[...,K-1],
                                        model.target: seq})[0]

        pred_raw = (inverse_transform(pred_raw[0].reshape([height, width, 3])) * 255.0).astype(np.uint8)

        pred_rgb = cv2.cvtColor(pred_raw, cv2.COLOR_YUV2RGB)
        cv2.imwrite(out, pred_rgb)


# Right now I just use two frame as the input
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--first", type=str, dest="first",
                        required=True, help="First reference image(T - 2)")
    parser.add_argument("--second", type=str, dest="second",
                        required=True, help="Second reference image(T - 1)")
    parser.add_argument("--out", type=str, dest="out",
                        required=True, help="Model to be used")                    

    args = parser.parse_args()
    main(**vars(args))
