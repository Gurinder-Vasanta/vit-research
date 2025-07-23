import chromadb
from chromadb import PersistentClient
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
    df = pd.read_csv('data/manual_intervals.csv')
    # print(df)

    output_dict = {}

    ls = np.array(df['left_start'])
    le = np.array(df['left_end'])

    left = []
    for i in range(len(ls)):
        try: 
            splitted = ls[i].split('_')
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
            none.append([ns[i],ne[i]])
        except: 
            continue
    output_dict['none'] = none
    return output_dict

layers = tf_keras.layers

chroma_client = PersistentClient(path='./chroma_store')

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

get_max_vid = lambda inter: int(inter[len(inter)-1][1].split('_')[0].split('vid')[1])

get_frame_num = lambda frm: int(frm.split('_')[2].split('.')[0])

get_vid_num = lambda frm: int(frm.split('_')[0].split('vid')[1])
max_vid = max(get_max_vid(list(im_ranges.values())[0]),
              get_max_vid(list(im_ranges.values())[1]),
              get_max_vid(list(im_ranges.values())[2]))
# input(max_vid)
# input(im_ranges)
# input(list(im_ranges.values())[0])
# input(list(im_ranges.values())[1])
# input(list(im_ranges.values())[2])


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
all_frames = sorted(all_frames, key = lambda frm: int(frm.split('_')[0].split('vid')[1]))
# input(all_frames)

total_count = 0
batch_cap = 1024 # was 32
cur_count = 0
embeddings = []
l_embeddings = []
r_embeddings = []
n_embeddings = []

# left_collection = chroma_client.get_or_create_collection(name="left_collection",metadata={"hnsw:space": "l2"})
# right_collection = chroma_client.get_or_create_collection(name="right_collection",metadata={"hnsw:space": "l2"})
# none_collection = chroma_client.get_or_create_collection(name="none_collection",metadata={"hnsw:space": "l2"})

embeddings = chroma_client.get_or_create_collection(name='embeddings',metadata={'hnsw:space': 'l2'})


l_fids = []
r_fids = [] 
n_fids = []

aux = []
aux_frame_ids = []
frame_ids = []

vid_to_frames_dict = {}
for f_name in all_frames: 
    vid_category = f_name.split('_')[0]
    if(vid_category not in vid_to_frames_dict.keys()):
        vid_to_frames_dict[vid_category] = [f_name]
    else:
        vid_to_frames_dict[vid_category].append(f_name)

# del(vid_to_frames_dict['vid1'])
# del(vid_to_frames_dict['vid2'])
# del(vid_to_frames_dict['vid3'])
# del(vid_to_frames_dict['vid4'])

vid = 'vid2'
# input(vid_to_frames_dict.keys())

directions = []
cur_fnames = []

for fname in vid_to_frames_dict[vid]: 
    frame_class = class_from_frame(fname)

    if(frame_class == 'ignore'): continue

    

    if(cur_count == batch_cap):

        aux = np.array(aux)
        print(aux.shape)
        print(len(directions))
        print(len(cur_fnames))
        output = model.predict(aux, batch_size=1024, verbose=1)
        temp = np.array(output['pre_logits'])
        temp = temp.reshape(batch_cap,hidden_size)
        print(temp)

        print(temp.shape)
        embeddings.add(
            embeddings=temp,
            ids=cur_fnames,
            metadatas = directions
        )

        aux = []
        cur_fnames = []
        directions = []
        cur_count = 0
    else:
        cur_count += 1
        cur_fnames.append(fname)
        target_size = (hidden_size,432)
        f_path = os.path.join(frames_path, fname)
        im = cv2.imread(f_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
        aux.append(temp_frame)
        directions.append({'label':frame_class})

if(len(aux) > 0):
    aux = np.array(aux)
    output = model.predict(aux, batch_size=len(aux), verbose=1)

    temp = np.array(output['pre_logits'])
    temp = temp.reshape(len(aux),hidden_size)

    embeddings.add(
        embeddings=temp,
        ids=cur_fnames,
        metadatas = directions
    )

    # input(frame_class)



    # target_size = (hidden_size,432)
    # temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
    # temp_frame = temp_frame.reshape(1,432,768,3)
    # # input(temp_frame.shape)
    # output = np.array(model.predict(temp_frame, batch_size = 1, verbose=1)['pre_logits'])
    # print(fname)
    # print(output[0][0][0])

    # if(frame_class == 'left'):
    #     left_collection.add(
    #         embeddings=output[0][0][0],
    #         ids=[fname[:-4]],
    #         metadatas = {'label':'left'}
    #     )
    #     # l_embeddings.append(temp[i])
    #     # l_fids.append(aux_frame_ids[i])
    #     # cv2.imwrite(f"{left_path}/left_{aux_frame_ids[i]}", im)
    # elif(frame_class == 'right'):
    #     right_collection.add(
    #         embeddings=output[0][0][0],
    #         ids=[fname[:-4]],
    #         metadatas = {'label':'right'}
    #     )
    # elif(frame_class == 'none'):
    #     none_collection.add(
    #         embeddings=output[0][0][0],
    #         ids=[fname[:-4]],
    #         metadatas = {'label':'none'}
    #     )



# for f_name in all_frames: 
#     # if(f_name.split('_')[0] == 'vid1'): continue
#     f_path = os.path.join(frames_path,f_name)
#     im = cv2.imread(f_path)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#     if(cur_count == batch_cap):
#         aux = np.array(aux)
#         output = model.predict(aux, batch_size = 1024, verbose=1) # batch_size was 32

#         temp = np.array(output['pre_logits'])
#         temp = temp.reshape(batch_cap,1,hidden_size)

#         for i in range(len(temp)):
#             # input(aux_frame_ids[i])
#             # input(aux_frame_ids)
#             f_class = class_from_frame(aux_frame_ids[i])

#             if(f_class == 'ignore'): continue
#             f_path = os.path.join(frames_path,aux_frame_ids[i])
#             im = cv2.imread(f_path)
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#             if(f_class == 'left'):
#                 l_embeddings.append(temp[i])
#                 l_fids.append(aux_frame_ids[i])
#                 cv2.imwrite(f"{left_path}/left_{aux_frame_ids[i]}", im)
#             elif(f_class == 'right'):
#                 r_embeddings.append(temp[i])
#                 r_fids.append(aux_frame_ids[i])
#                 cv2.imwrite(f"{right_path}/right_{aux_frame_ids[i]}", im)
#             elif(f_class == 'none'):
#                 n_embeddings.append(temp[i])
#                 n_fids.append(aux_frame_ids[i])
#                 cv2.imwrite(f"{none_path}/none_{aux_frame_ids[i]}", im)

#         # for embd in temp:
#         #     f_class = class_from_frame()
#         #     embeddings.append(embd)
        
#         # frame_ids = np.append(np.array(frame_ids),np.array(aux_frame_ids))

#         aux = []
#         aux_frame_ids = []
#         cur_count = 0
    
#     else:
#         cur_count += 1
#         target_size = (hidden_size,432)
#         temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
#         aux.append(temp_frame)
#         aux_frame_ids.append(f_name)
# for vid_num in range(1,max_vid+1):
#     for f_name in all_frames: 
#         f_path = os.path.join(frames_path, f_name)
#         im = cv2.imread(f_path)
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# cur_vid_num = 1

# all_frames=['vid1_frame_1996.jpg','vid2_frame_553.jpg','vid3_frame_3200.jpg','vid4_frame_3974.jpg']
# for f_name in all_frames: 
#     input(f_name)
#     # input(f_name)
#     # input(f_name.split('_')[0].split('vid'))
#     vid_num = get_vid_num(f_name)
#     if(vid_num > cur_vid_num):
#         print(f'vid{cur_vid_num} info')
#         print(np.array(l_embeddings).shape)
#         print(np.array(r_embeddings).shape)
#         print(np.array(n_embeddings).shape)
#         print(np.array(l_fids).shape)
#         print(np.array(r_fids).shape)
#         print(np.array(n_fids).shape)

#         np.savez(f'data/embeddings/vid{cur_vid_num}_left_embeddings.npz',embeddings=l_embeddings,frame_ids=l_fids)
#         np.savez(f'data/embeddings/vid{cur_vid_num}_right_embeddings.npz',embeddings=r_embeddings,frame_ids=r_fids)
#         np.savez(f'data/embeddings/vid{cur_vid_num}_none_embeddings.npz',embeddings=n_embeddings,frame_ids=n_fids)
#         cur_vid_num = vid_num

#         total_count = 0
#         batch_cap = 1024 # was 32
#         cur_count = 0
#         embeddings = []
#         l_embeddings = []
#         r_embeddings = []
#         n_embeddings = []

#         l_fids = []
#         r_fids = [] 
#         n_fids = []

#         aux = []
#         aux_frame_ids = []
#         frame_ids = []
#     # input(vid_num)
#     print('hello??')
#     f_path = os.path.join(frames_path,f_name)
#     im = cv2.imread(f_path)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#     if(cur_count == batch_cap):
#         aux = np.array(aux)
#         output = model.predict(aux, batch_size = 1024, verbose=1) # batch_size was 32

#         temp = np.array(output['pre_logits'])
#         temp = temp.reshape(batch_cap,1,hidden_size)

#         for i in range(len(temp)):
#             f_class = class_from_frame(aux_frame_ids[i])

#             if(f_class == 'ignore'): continue
#             f_path = os.path.join(frames_path,aux_frame_ids[i])
#             im = cv2.imread(f_path)
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#             if(f_class == 'left'):
#                 l_embeddings.append(temp[i])
#                 l_fids.append(aux_frame_ids[i])
#                 cv2.imwrite(f"{left_path}/left_{aux_frame_ids[i]}", im)
#             elif(f_class == 'right'):
#                 r_embeddings.append(temp[i])
#                 r_fids.append(aux_frame_ids[i])
#                 cv2.imwrite(f"{right_path}/right_{aux_frame_ids[i]}", im)
#             elif(f_class == 'none'):
#                 n_embeddings.append(temp[i])
#                 n_fids.append(aux_frame_ids[i])
#                 cv2.imwrite(f"{none_path}/none_{aux_frame_ids[i]}", im)

#         aux = []
#         aux_frame_ids = []
#         cur_count = 0
    
#     else:
#         cur_count += 1
#         target_size = (hidden_size,432)
#         temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
#         aux.append(temp_frame)
#         aux_frame_ids.append(f_name)



# np.savez('data/embeddings/left_embeddings.npz',embeddings=l_embeddings,frame_ids=l_fids)
# np.savez('data/embeddings/right_embeddings.npz',embeddings=r_embeddings,frame_ids=r_fids)
# np.savez('data/embeddings/none_embeddings.npz',embeddings=n_embeddings,frame_ids=n_fids)
model.save_weights('vit_random_weights.h5')

