from official.vision.modeling.backbones import vit
import tensorflow as tf
import numpy as np
import tensorflow as tf, tf_keras
import skvideo.io
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# need to figure out how to manage multiple videos. possible options: 
# 1. before each frame, put vid2, vid3, etc
# considerations: 
# so far, everything is built on a frame_ syntax. going to need to change that
# manual_intervals also only keeps track of numbers, so either: 
# we need to make a whole new .csv file for each video or 
# encode which video it is within the interval itself (so do it like v2_103 as left_start and v2_402 as left_end)
# this wouldn't be too bad. you still keep track of the number and you can get the video from the frame name

# 2. just start at the last entered frame number (so 24750) and add it as frame_24751 etc
# considerations: 
# then you can't tell what frame the video came from so its harder to know what full possession the frame belongs to


layers = tf_keras.layers
# print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
# from tensorflow.python.platform import build_info as tf_build_info
# print(tf_build_info.cuda_version_number)
# print(tf_build_info.cudnn_version_number)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '3' to suppress everything
  
#1080 rows
#1920 cols
#3 channels
inputparameters = {}
outputparameters = {}

output_path = 'data/raw_videos/heat_2014.mp4'
# reader = skvideo.io.FFmpegReader(output_path,
#                                  inputdict=inputparameters,
#                                  outputdict=outputparameters)

 
 
hidden_size = 768 #768
model = vit.VisionTransformer(
    # image_size = 224,
    input_specs=layers.InputSpec(shape=[None,432,768,3]),  #432,768   648,1152   504,896
    patch_size=256, #16 128 had more variance 64
    num_layers=12, #12  14
    num_heads=12, #12  14
    hidden_size=hidden_size,
    mlp_dim=3072 #3072  3584
)

cap = cv2.VideoCapture(output_path)
batch_cap = 256
cur_count = 0
embeddings = []
aux = []
frames = []

max_frames = 12000 #10240  12288

global_counter = 0

# this file should basically be just for manually labelling frames (no real use for it after that)
embeddings_count = 0
while True:
    if(global_counter == max_frames): break #10240
    ret, frame = cap.read()
    if not ret: break
    global_counter+=1
    print(global_counter)
    if(global_counter <= 6000): continue
    # print('ec then gc')
    # print(embeddings_count)
    # if(global_counter <= 1540): continue
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    frames.append(frame)
    cur_count += 1
    target_size = (hidden_size,432) #504
    # first write all frames to ims so that i can just pull what frames I need after
    cv2.imwrite(f"data/unseen_test_images/ims/vid3_frame_{global_counter}.jpg", frame)
    # temp_frame = cv2.resize(frame,target_size,interpolation=cv2.INTER_AREA)
    
    # aux.append(temp_frame)
# input('stop')
