
# coding: utf-8

# In[19]:


import os
import numpy as np
import pandas as pd
import cv2
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


# In[20]:


cwd = os.getcwd()
data_dir = "C:\\Users\\windo\\Desktop\\beta_simulator_windows\\New folder"
log_file_dir = os.path.join(data_dir, "driving_log.csv")
img_dir = os.path.join(data_dir, "IMG")


# In[21]:


batch_size = 32
img_h = 66
img_w = 200
num_channels = 3
bias = 0.5
delta_correlation = 0.25


# In[22]:


# os.listdir(data_dir)


# In[70]:


def split_data(test_size=0.2):
    cols = ["left", "center", "right", "steer", "throttle", "brake", "speed"]
    csv = pd.read_csv(log_file_dir, names=cols)
    csv = csv.values
    train_data, validation_data = train_test_split(csv, test_size=test_size)
    return train_data, validation_data


# In[71]:


def preprocess_img(img_path):
    img = cv2.imread(img_path)
    cropped_img = img[range(20, 120), :, :]
    resized_img = cv2.resize(cropped_img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    return resized_img.astype('float32')

def load_batch(data, batch_size=batch_size, augment_data=True, bias=0.5):
    x = np.zeros(shape=(1, img_h, img_w, num_channels), dtype=np.float32)
    y_steer = np.zeros(shape=(1), dtype=np.float32)
    
    while len(x) <= batch_size:
        idx = np.random.choice(len(data))
        ct_path, lt_path, rt_path, steer, throttle, brake, speed = data[idx]
        steer = np.float32(steer)
        cam_choice = np.random.choice(['center', 'left', 'right'])
        if cam_choice == 'center':
            img_path = ct_path
            steer = steer
        elif cam_choice == 'left':
            img_path = lt_path
            steer = steer + delta_correlation
        elif cam_choice == 'right':
            img_path = rt_path
            steer = steer - delta_correlation
        
        if (abs(steer) + bias) < np.random.rand():
            pass
    
        img = preprocess_img(img_path)    
        if augment_data:
            
            if random.choice([True, False]):
                img = img[:, ::-1, :]
                steer *= -1.
            
            #change brightness randomly
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 2] *= random.uniform(a=0.2, b=1.5)
            img[:, :, 2] = np.clip(img[:, :, 2], a_min=0, a_max=255)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        img = np.expand_dims(img, 0)
        x = np.vstack([x, img])
        y_steer = np.hstack([y_steer, steer])
        
    return x[1:], y_steer[1:]

def generator(data, batch_size=batch_size, augment_data=True, bias=0.5):
    while True:
        x, y = load_batch(data, batch_size, augment_data, bias)
        yield x, y


# In[76]:


# t, v = split_data()
# x, y = next(generator(t))
# print(x.shape, y.shape)

