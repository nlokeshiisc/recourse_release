# %%
# !pip install imagecorruptions

# %%
import pickle as pickle
import os
import numpy as np
import cv2 
#%matplotlib inline
import matplotlib.pyplot as plt
import sys

def load_data(file_name):
    assert(os.path.exists(file_name+'.pkl'))
    with open(file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# %%
data  = load_data(sys.argv[1])

# %%
img = data[2][0]
#cv2.imshow('a',img)
#img = np.asarray(img, np.float64)

plt.imshow(np.transpose(img,(1,2,0)))

# %%
from imagecorruptions import corrupt, get_corruption_names
corrupted_imgs = []
for name in get_corruption_names():
    corrupted_image = corrupt(np.transpose(img.astype(np.uint8), (1, 2, 0)), corruption_name=name, severity=np.random.randint(5, 6))
    corrupted_imgs.append(corrupted_image)

_, axs = plt.subplots(3, 5, figsize=(12, 12))
axs = axs.flatten()
for img, ax in zip(corrupted_imgs, axs):
    ax.imshow(img)
plt.show()

# %%
def add_gaussian_noise(img):
    img = img.astype('uint8')
    gauss = np.random.normal(0,0.5,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img,gauss)
    return img_gauss

def add_sk_noise(img):
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    noise = img + img * gauss
    return noise    

# %%
# tensor([0, 0, 3]) : [0.95, 0.88, 0.66, 0.77, 0.76, 0.76, 0.79, 0.85, 0.71, 0.71]
# tensor([0, 1, 0]) : [0.96, 0.88, 0.94, 0.89, 0.82, 0.85, 0.93, 0.89, 0.73, 0.9]
# tensor([1, 0, 1]) : [0.95, 0.66, 0.91, 0.71, 0.84, 0.62, 0.9, 0.77, 0.2, 0.89]
# tensor([2, 0, 2]) : [0.91, 0.68, 0.95, 0.68, 0.68, 0.73, 0.88, 0.74, 0.62, 0.96]
# tensor([2, 1, 0]) : [0.96, 0.69, 0.96, 0.95, 0.73, 0.79, 0.96, 0.84, 0.71, 0.93]
# tensor([3, 1, 0]) : [0.98, 0.91, 0.99, 0.8, 0.9, 0.93, 0.93, 0.85, 0.79, 0.91]
# tensor([3, 2, 1]) : [0.95, 0.95, 0.98, 0.81, 0.81, 0.89, 0.9, 0.89, 0.69, 0.99]
# tensor([4, 2, 2]) : [0.96, 0.79, 0.96, 0.88, 0.82, 0.74, 0.93, 0.84, 0.62, 0.94]
# tensor([5, 2, 3]) : [0.94, 0.81, 0.96, 0.77, 0.86, 0.88, 0.99, 0.86, 0.68, 0.99]

# %%
import random
from imagecorruptions import get_corruption_names, corrupt
corruption_types = get_corruption_names()

corruption_types = [corruption_types[entry] for entry in [0, 1, 2, 8, 9, 10, 12, 13, 14]]
print(corruption_types)

def do_corruption(x):
     x = x.astype(np.uint8)
     corrupted_image = corrupt(np.transpose(x, (1, 2, 0)), corruption_name=np.random.choice(corruption_types), severity=np.random.randint(5, 6))
     return np.transpose(corrupted_image, (2, 0, 1))

data_with_noise = []
for tup in data:
    x = tup[0]
    y = tup[3]
    beta = tup[2] 
    p = random.uniform(0,1)
    if p < 0.8:
     if beta==[0,0,3] and y in [0,1,7]:
          x = do_corruption(x)
     if beta==[0,1,0] and y in [1, 2, 6, 9]:
          x = do_corruption(x)
     if beta==[1,0,1] and y in [0, 9]:
          x = do_corruption(x)
     if beta==[2,0,2] and y in [0, 7, 9]:
          x = do_corruption(x)
     if beta==[2,1,0] and y in [0]:
          x = do_corruption(x)
     if beta==[3,1,0] and y in [2,7,9]:
          x = do_corruption(x)
     if beta==[3,2,1] and y in [0,1,2,7,9]:
          x = do_corruption(x)
     if beta==[4,2,2] and y in [0,2,7,9]:
          x = do_corruption(x)
     if beta==[5,2,3] and y in [0, 1, 2, 3, 6]:
          x = do_corruption(x)
    tup[0] = x
    data_with_noise.append(tup)
print(len(data_with_noise))

# %%
_, axs = plt.subplots(3, 5, figsize=(12, 12))
axs = axs.flatten()
for img, ax in zip(data_with_noise[0:15], axs):
    print(img[0].shape)
    ax.imshow(np.transpose(img[0], (1, 2, 0)))
plt.show()

# %%
with open(sys.argv[2],"wb") as file:
    pickle.dump(data_with_noise,file)

# %%



