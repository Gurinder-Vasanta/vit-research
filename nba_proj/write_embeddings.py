from official.vision.modeling.backbones import vit
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf, tf_keras
import skvideo.io
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os

def generate_manual_intervals():
    # df = pd.read_csv('data/manual_intervals.csv')
    # # print(df)

    # output_dict = {}

    # ls = np.array(df['left_start'])
    # le = np.array(df['left_end'])

    # left = []
    # for i in range(len(ls)):
    #     if(not np.isnan(ls[i])):
    #         left.append([int(ls[i]),int(le[i])])
    # output_dict['left'] = left

    # rs = np.array(df['right_start'])
    # re = np.array(df['right_end'])

    # right = []
    # for i in range(len(rs)):
    #     if(not np.isnan(rs[i])):
    #         right.append([int(rs[i]),int(re[i])])
    # output_dict['right'] = right

    # ns = np.array(df['none_start'])
    # ne = np.array(df['none_end'])

    # none = []
    # for i in range(len(ns)):
    #     if(not np.isnan(ns[i])):
    #         none.append([int(ns[i]),int(ne[i])])
    # output_dict['none'] = none
    # return output_dict
    df = pd.read_csv('data/manual_intervals.csv')
    # print(df)

    output_dict = {}

    ls = np.array(df['left_start'])
    le = np.array(df['left_end'])

    left = []
    for i in range(len(ls)):
        try: 
            splitted = ls[i].split('_')
            if(splitted[0] == 'vid1'): continue # temporarily skip vid1 (vid1 is not a full game)
            left.append([ls[i],le[i]])
        except: 
            continue
    output_dict['left'] = left

    rs = np.array(df['right_start'])
    re = np.array(df['right_end'])

    right = []
    for i in range(len(rs)):
        try: 
            splitted = rs[i].split('_')
            if(splitted[0] == 'vid1'): continue # temporarily skip vid1 (vid1 is not a full game)
            right.append([rs[i],re[i]])
        except: 
            continue
    output_dict['right'] = right

    ns = np.array(df['none_start'])
    ne = np.array(df['none_end'])

    none = []
    for i in range(len(ns)):
        try: 
            splitted = ns[i].split('_')
            if(splitted[0] == 'vid1'): continue # temporarily skip vid1 (vid1 is not a full game)
            none.append([ns[i],ne[i]])
        except: 
            continue
    output_dict['none'] = none
    return output_dict

layers = tf_keras.layers

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

# going to add more intervals to this if the embeddings are still bad
# maybe put this in a separate text file instead of manually hardcoding dictionary
# im_ranges = {'left':[[1,420],[1304,1842],[1872,2056],[2161,2414]],
#             'right':[[458,896],[954,1303],[2464,2650],[2671,2879],[2949,3198],[3292,3540]],
#             'none':[[421,458],[897,953],[1843,1871],[2057,2160],[2415,2463],[2651,2670],[2880,2948],[3199,3291],[3541,3595]]}

im_ranges = generate_manual_intervals()

def class_from_frame(frame_name):
    splitted = frame_name.split('_')
    # input(splitted)
    # input(im_ranges['left'])
    num = int(splitted[2].split('.')[0])
    # input(num)
    if(splitted[0] == 'vid3' and num <= 4900 and num >= 1): return 'ignore' # limit the number of none frames
    for inter in im_ranges['left']:
        # print('in ranges left')
        # input(inter)
        # inter: ['vid1_1', 'vid1_420']
        vid_str = inter[0].split('_')[0]
        inter_start = int(inter[0].split('_')[1])
        inter_end = int(inter[1].split('_')[1])
        if(num >= inter_start and num <= inter_end and splitted[0] == vid_str): return 'left'
    for inter in im_ranges['right']:
        # input(inter)
        # print('in ranges left')
        # input(inter)
        # inter: ['vid1_1', 'vid1_420']
        vid_str = inter[0].split('_')[0]
        inter_start = int(inter[0].split('_')[1])
        inter_end = int(inter[1].split('_')[1])
        if(num >= inter_start and num <= inter_end and splitted[0] == vid_str): return 'right'
    # input(frame_name)
    return 'none'

frames_path = 'data/temp'
left_path = 'data/left'
right_path = 'data/right'
none_path = 'data/none'

all_frames = os.listdir(frames_path)

total_count = 0
batch_cap = 1024 # was 32
cur_count = 0
embeddings = []
l_embeddings = []
r_embeddings = []
n_embeddings = []

l_fids = []
r_fids = [] 
n_fids = []

aux = []
aux_frame_ids = []
frame_ids = []

for f_name in all_frames: 
    # if(f_name.split('_')[0] == 'vid1'): continue
    f_path = os.path.join(frames_path,f_name)
    im = cv2.imread(f_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if(cur_count == batch_cap):
        aux = np.array(aux)
        output = model.predict(aux, batch_size = 1024, verbose=1) # batch_size was 32

        temp = np.array(output['pre_logits'])
        temp = temp.reshape(batch_cap,1,hidden_size)

        for i in range(len(temp)):
            # input(aux_frame_ids[i])
            # input(aux_frame_ids)
            f_class = class_from_frame(aux_frame_ids[i])

            if(f_class == 'ignore'): continue
            f_path = os.path.join(frames_path,aux_frame_ids[i])
            im = cv2.imread(f_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if(f_class == 'left'):
                l_embeddings.append(temp[i])
                l_fids.append(aux_frame_ids[i])
                cv2.imwrite(f"{left_path}/left_{aux_frame_ids[i]}", im)
            elif(f_class == 'right'):
                r_embeddings.append(temp[i])
                r_fids.append(aux_frame_ids[i])
                cv2.imwrite(f"{right_path}/right_{aux_frame_ids[i]}", im)
            elif(f_class == 'none'):
                n_embeddings.append(temp[i])
                n_fids.append(aux_frame_ids[i])
                cv2.imwrite(f"{none_path}/none_{aux_frame_ids[i]}", im)

        # for embd in temp:
        #     f_class = class_from_frame()
        #     embeddings.append(embd)
        
        # frame_ids = np.append(np.array(frame_ids),np.array(aux_frame_ids))

        aux = []
        aux_frame_ids = []
        cur_count = 0
    
    else:
        cur_count += 1
        target_size = (hidden_size,432)
        temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
        aux.append(temp_frame)
        aux_frame_ids.append(f_name)

# print(np.array(embeddings).shape)
# print(np.array(frame_ids).shape)

print(np.array(l_embeddings).shape)
print(np.array(r_embeddings).shape)
print(np.array(n_embeddings).shape)
print(np.array(l_fids).shape)
print(np.array(r_fids).shape)
print(np.array(n_fids).shape)

np.savez('data/embeddings/left_embeddings.npz',embeddings=l_embeddings,frame_ids=l_fids)
np.savez('data/embeddings/right_embeddings.npz',embeddings=r_embeddings,frame_ids=r_fids)
np.savez('data/embeddings/none_embeddings.npz',embeddings=n_embeddings,frame_ids=n_fids)
model.save_weights('vit_random_weights.h5')

# print(embeddings[0])
# print(frame_ids[0])
# print(class_from_frame(frame_ids[0]))


# the left right none folders should have the frames themselves and the corresponding embeddings files

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

