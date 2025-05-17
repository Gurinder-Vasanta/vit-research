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
import os

inputparameters = {}
outputparameters = {}

output_path = 'data/hoop_ball_finding.mp4'

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

# these are labelled frames (put the manually labelled ones in the temp folder)
# 1-420 is left
# 421 to 458 is none
# 458 to 896 is right
# 897 to 953 is none
# 954 to 1303 is right
# 1304 to 1540+ is left

im_ranges = {'left':[[1,420],[1304,1540]],'right':}
frames_path = 'data/temp'
all_frames = os.listdir(frames_path)

batch_cap = 256
cur_count = 0
embeddings = []
aux = []

for f_name in all_frames: 
    f_path = os.path.join(frames_path,f_name)
    im = cv2.imread(f_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# cap = cv2.VideoCapture(output_path)
# batch_cap = 256
# cur_count = 0
# embeddings = []
# aux = []
# frames = []

# max_frames = 1024 #10240  12288
# train_ind,test_ind = train_test_split([i for i in range(max_frames)],train_size=0.8,random_state=0)


# global_counter = 0
# while True:
#     if(len(embeddings) == max_frames): break #10240
#     ret, frame = cap.read()
#     if not ret: break
#     global_counter+=1
#     if(global_counter < 514): continue
#     frames.append(frame)
#     if(cur_count == batch_cap):
#         aux = np.array(aux)
#         # input(aux.shape)
    
#         output = model.predict(aux, batch_size = 32, verbose=1)
#         aux = []
#         cur_count = 0
#         # print(output)
#         print(output['pre_logits'].shape) # generates a 768 length vector for each frame, so the shape is 256,1,1,768
#         temp = np.array(output['pre_logits'])
#         temp = temp.reshape(batch_cap,1,hidden_size)
#         for embd in temp:
#             embeddings.append(embd)
#         print(np.array(embeddings).shape)
#     else:
#         cur_count += 1
#         target_size = (hidden_size,432) #504
#         cv2.imwrite(f"data/temp/frame_{global_counter}.jpg", frame)
#         temp_frame = cv2.resize(frame,target_size,interpolation=cv2.INTER_AREA)
        
#         aux.append(temp_frame)

