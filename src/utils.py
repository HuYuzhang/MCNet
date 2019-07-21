"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import numpy as np
import IPython

def transform(image):
  """
  Function transform will trans a (0-255) picture to a (-1.0, 1.0) picture
  """
  return image/127.5 - 1.


def inverse_transform(images):
  """
  Function inverse_transform will trans a (-1.0, 1.0) picture to a (0.0, 1.0) picture
  """
  return (images+1.)/2.


def save_images(images, size, image_path):
  return imsave(inverse_transform(images)*255., size, image_path)


def merge(images, size, YUV=True):
  size = [int(x) for x in size]
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3), dtype=np.uint8)
  images = images.astype(np.uint8)
  for idx, image in enumerate(images):
    i = int(idx % size[1])
    j = int(idx / size[1])
    if YUV:
      # IPython.embed()
      image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
      # IPython.embed()
      # exit(0)
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  img = img.astype(np.uint8)
  return img


def imsave(images, size, path):
  tmp_img = merge(images, size)
  tmp_img = tmp_img.astype(np.uint8)
  return cv2.imwrite(path, tmp_img)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    # Make a minibatch out of what is left
    # And this batch is samller than normal batch
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 


def load_kth_data(f_name, data_path, image_size, K, T): 
  flip = np.random.binomial(1,.5,1)[0]
  tokens = f_name.split()
  vid_path = data_path + tokens[0] + "_uncomp.avi"
  vid = imageio.get_reader(vid_path,"ffmpeg")
  low = int(tokens[1])
  high = np.min([int(tokens[2]),vid.get_length()])-K-T+1
  if low == high:
    stidx = 0 
  else:
    if low >= high: print(vid_path)
    stidx = np.random.randint(low=low, high=high)
  seq = np.zeros((image_size, image_size, K+T, 1), dtype="float32")
  for t in range(K+T):
    try:
      img = cv2.cvtColor(cv2.resize(vid.get_data(stidx+t),
                        (image_size,image_size)),
                        cv2.COLOR_RGB2GRAY)
    except Exception:
      print("Error detect by hyz: in utils.py: line 113")
      import IPython
      IPython.embed()
    seq[:,:,t] = transform(img[:,:,None])

  if flip == 1:
    seq = seq[:,::-1]

  diff = np.zeros((image_size, image_size, K-1, 1), dtype="float32")
  for t in range(1,K):
    prev = inverse_transform(seq[:,:,t-1])
    next = inverse_transform(seq[:,:,t])
    diff[:,:,t-1] = next.astype("float32")-prev.astype("float32")

  return seq, diff


def load_vimeo_data(f_name, data_path, image_size, K, T):
  seq = np.zeros((image_size, image_size, K+T, 3), dtype="float32")
  for t in range(0, K+T):
    # print(data_path  + f_name + "/im%d.png"%(t + 1))
    img = cv2.imread(data_path  + f_name + "/im%d.png"%(t + 1))
    # Below is the part for adapt the shape of image to right shape, right now I just take a simple indice
    # Here our input's cahnnel is 3, for YUV each
    # IPython.embed()
    try:
      yuv_img = cv2.cvtColor(img[:image_size, :image_size, :], cv2.COLOR_RGB2YUV)
    except Exception:
      IPython.embed()
    seq[:,:,t] = transform(yuv_img)
  # Then cal the diff, though we only have one diff
  diff = np.zeros((image_size, image_size, K-1, 3), dtype="float32")
  for t in range(1, K):
      prev = inverse_transform(seq[:,:,t-1])
      cur  = inverse_transform(seq[:,:,t  ])
      diff[:,:,t - 1] = cur.astype("float32") - prev.astype("float32")
  return seq, diff

def load_s1m_data(f_name, data_path, trainlist, K, T):
  flip = np.random.binomial(1,.5,1)[0]
  vid_path = data_path + f_name
  img_size = [240,320]

  while True:
    try:
      vid = imageio.get_reader(vid_path,"ffmpeg")
      low = 1
      high = vid.get_length()-K-T+1
      if low == high:
        stidx = 0
      else:
        stidx = np.random.randint(low=low, high=high)
      seq = np.zeros((img_size[0], img_size[1], K+T, 3),
                     dtype="float32")
      for t in range(K+T):
        img = cv2.resize(vid.get_data(stidx+t),
                         (img_size[1],img_size[0]))[:,:,::-1]
        seq[:,:,t] = transform(img)

      if flip == 1:
        seq = seq[:,::-1]

      diff = np.zeros((img_size[0], img_size[1], K-1, 1),
                      dtype="float32")
      for t in range(1,K):
        prev = inverse_transform(seq[:,:,t-1])*255
        prev = cv2.cvtColor(prev.astype("uint8"),cv2.COLOR_BGR2GRAY)
        next = inverse_transform(seq[:,:,t])*255
        next = cv2.cvtColor(next.astype("uint8"),cv2.COLOR_BGR2GRAY)
        diff[:,:,t-1,0] = (next.astype("float32")-prev.astype("float32"))/255.
      break
    except Exception:
      # In case the current video is bad load a random one 
      rep_idx = np.random.randint(low=0, high=len(trainlist))
      f_name = trainlist[rep_idx]
      vid_path = data_path + f_name

  return seq, diff

