import cv2
import sys
import time
import imageio
import os
import IPython

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed


def main(lr, batch_size, image_size, K,
        T, num_iter, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu[0])
    
    data_path = "../data/vimeo_interp_test/target/"
    f = open(data_path+"tri_testlist.txt","r")
    trainfiles = [l[:-1] for l in f.readlines()]
    margin = 0.3 
    updateD = False
    updateG = True
    iters = 0
    prefix  = ("VIMEO_MCNET"
            + "_image_size="+str(image_size)
            + "_K="+str(K)
            + "_T="+str(T)
            + "_batch_size="+str(batch_size)
            + "_lr="+str(lr))

    print("\n"+prefix+"\n")
    checkpoint_dir = "../models/"+prefix+"/"
    samples_dir = "../samples/"+prefix+"/"
    summary_dir = "../logs/"+prefix+"/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(summary_dir):
        makedirs(summary_dir)
    
    with tf.device("/gpu:%d"%gpu[0]):
        model = MCNET(image_size=[image_size,image_size], c_dim=3,
                    K=K, batch_size=batch_size, T=T,
                    checkpoint_dir=checkpoint_dir)

        g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            model.L_img, var_list=model.g_vars
        )

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                    log_device_placement=False,
                    gpu_options=gpu_options)) as sess:

        tf.global_variables_initializer().run()

        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        g_sum = tf.summary.merge([model.L_p_sum,
                                model.L_gdl_sum, model.loss_sum])

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters+1
        start_time = time.time()
        while iters < num_iter:
            mini_batches = get_minibatches_idx(len(trainfiles), batch_size, shuffle=True)
            for _, batchidx in mini_batches:
                if len(batchidx) == batch_size:
                    seq_batch  = np.zeros((batch_size, image_size, image_size,
                                            K+T, 3), dtype="float32")
                    diff_batch = np.zeros((batch_size, image_size, image_size,
                                            K-1, 3), dtype="float32")
                    t0 = time.time()
                    Ts = np.repeat(np.array([T]),batch_size,axis=0)
                    Ks = np.repeat(np.array([K]),batch_size,axis=0)
                    paths = np.repeat(data_path, batch_size,axis=0)
                    tfiles = np.array(trainfiles)[batchidx]
                    shapes = np.repeat(np.array([image_size]),batch_size,axis=0)
                    output = [load_vimeo_data(f, p,img_sze, k, t)
                                                            for f,p,img_sze,k,t in zip(tfiles,
                                                                                    paths,
                                                                                    shapes,
                                                                                    Ks, Ts)]

                for i in range(batch_size):
                    seq_batch[i] = output[i][0]
                    diff_batch[i] = output[i][1]


                _, summary_str = sess.run([g_optim, g_sum],
                                            feed_dict={model.diff_in: diff_batch,
                                                    model.xt: seq_batch[:,:,:,K-1],
                                                    model.target: seq_batch})
                writer.add_summary(summary_str, counter)


                errL_img = model.L_img.eval({model.diff_in: diff_batch,
                                            model.xt: seq_batch[:,:,:,K-1],
                                            model.target: seq_batch})
                counter += 1
                if counter % 50 == 0:
                    print(
                        "Iters: [%2d] time: %4.4f, L_img: %.8f" 
                        % (iters, time.time() - start_time, errL_img)
                    )

                if np.mod(counter, 10) == 1:
                    samples = sess.run([model.G],
                                        feed_dict={model.diff_in: diff_batch,
                                                    model.xt: seq_batch[:,:,:,K-1],
                                                    model.target: seq_batch})[0]
                    # IPython.embed()
                    samples = samples[0].swapaxes(0,2).swapaxes(1,2)
                    # IPython.embed()

                    sbatch  = seq_batch[0,:,:,:].swapaxes(0,2).swapaxes(1,2)
                    
                    sbatch2 = sbatch.copy()
                    # IPython.embed()
                    sbatch2[K:,:,:] = samples
                    # IPython.embed()
                    samples = np.concatenate((sbatch2,sbatch), axis=0)
                    # IPython.embed()
                    print("Saving sample ...")
                    save_images(samples, [2, K+T], 
                                samples_dir+"train_%s.png" % (iters))
                if np.mod(counter, 10000) == 2:
                    model.save(sess, checkpoint_dir, counter)

                iters += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.0001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=8, help="Mini-batch size")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=256, help="Mini-batch size")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")

    args = parser.parse_args()
    main(**vars(args))
