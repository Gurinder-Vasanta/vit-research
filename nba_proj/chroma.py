import chromadb
from chromadb import PersistentClient

from official.vision.modeling.backbones import vit
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

import tensorflow as tf, tf_keras
import cv2
import numpy as np
import os
from joblib import load

top_n_closest = 20

def determine_class_pass1(ids, metadatas, distances):
    # some of them are pretty unconfident and wrong
    # like some of the frames have the lowest distance at like 150
    # a pretty good distance is aorund 60, so maybe we should be doing this in 2 passes
    # first pass handles distances between 60 and 100
    #     if 80% (arbitrarily decided) or more agree, then thats the class and add the embeddng to the chromadb
    #     if its not 80%, then do the weighted distances thing
    # second pass does the same thing as the first pass, but just with more embeddings in the chromadb

    num_results = len(ids)
    num_left = 0
    num_right = 0
    num_none = 0
    for i in range(num_results): 
        if(metadatas[i]['label'] == 'left'):
            num_left += 1
        elif(metadatas[i]['label'] == 'right'):
            num_right += 1
        else:
            num_none += 1
    print(num_left)
    print(num_right)
    print(num_none)

    nums = [num_left, num_right, num_none]
    if(max(nums)>=17):
        a = np.array(nums).argmax()
        if(a == 0):
            return 'left'
        elif(a == 1):
            return 'right'
        else:
            return 'none'
    else:
        return 'pass2'
    # calcd_distances = []
    # calcd_left = []
    # calcd_right = []
    # calcd_none = []

    # # simply cant be num_left/num_results * distances because
    # # if our goal is to get minimum distances, then if theres like 3 wrong ones, those values would be super low

    # for i in range(num_results):
    #     if(metadatas[i]['label'] == 'left'):
    #         calcd_distances.append(((num_results - num_left)/num_results) * distances[i])
    #         calcd_left.append((num_left/num_results) * distances[i])
    #     elif(metadatas[i]['label'] == 'right'):
    #         calcd_distances.append(((num_results - num_right)/num_results) * distances[i])
    #         calcd_right.append((num_right/num_results) * distances[i])
    #     else: 
    #         calcd_distances.append(((num_results - num_none)/num_results) * distances[i])
    #         calcd_none.append((num_none/num_results) * distances[i])


    # print(calcd_distances)
    # print()
    # print(calcd_left)
    # print()
    # print(calcd_right)
    # print()
    # print(calcd_none)

    # print(np.mean(np.array(calcd_left)))

def determine_class_pass2(ids, metadatas, distances):
    # some of them are pretty unconfident and wrong
    # like some of the frames have the lowest distance at like 150
    # a pretty good distance is aorund 60, so maybe we should be doing this in 2 passes
    # first pass handles distances between 60 and 100
    #     if 80% (arbitrarily decided) or more agree, then thats the class and add the embeddng to the chromadb
    #     if its not 80%, then do the weighted distances thing
    # second pass does the same thing as the first pass, but just with more embeddings in the chromadb

    num_results = len(ids)
    num_left = 0
    num_right = 0
    num_none = 0
    for i in range(num_results): 
        if(metadatas[i]['label'] == 'left'):
            num_left += 1
        elif(metadatas[i]['label'] == 'right'):
            num_right += 1
        else:
            num_none += 1
    print(num_left)
    print(num_right)
    print(num_none)

    nums = [num_left, num_right, num_none]
    
    a = np.array(nums).argmax()
    if(a == 0):
        return 'left'
    elif(a == 1):
        return 'right'
    else:
        return 'none'

    


vid = 'vid3'

layers = tf_keras.layers

client = PersistentClient(path="./chroma_store")

embeddings = client.get_or_create_collection(name=f"{vid}_embeddings",metadata={"hnsw:space": "l2"})

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


model.load_weights('vit_random_weights.h5')
# wrongly classified as none (supposed to be left):
# /home/vasantgc/venv/nba_proj/data/unseen_test_images/none/none_vid2_frame_11168.jpg
# 20 closest vectors were indeed left

# wrongly classified as right (supposed to be left): 
# /home/vasantgc/venv/nba_proj/data/unseen_test_images/right/right_vid2_frame_11167.jpg

# vid 4 image: 
# /home/vasantgc/venv/nba_proj/data/unseen_test_images/ims/vid4_frame_5322.jpg

# combining all videos into 1 collection is not working at all 
# likely because some none frames have a lot of white so the distances are minimizing for some reason

# vid 3 frame (should be left): 
# /home/vasantgc/venv/nba_proj/data/unseen_test_images/ims/vid3_frame_11964.jpg
# create separate collections for each video

predicted_lefts = []
predicted_rights = []
predicted_nones = []

predicted_l_ids = []
predicted_r_ids = []
predicted_n_ids = []

pass2s = []
pass2_ids = []

unseens = '/home/vasantgc/venv/nba_proj/data/unseen_test_images/ims'
frames = os.listdir('/home/vasantgc/venv/nba_proj/data/unseen_test_images/ims')

# input(frames)
temp = []
cur_fnames = []
directions = []

for image in frames: 
    if(image.split('_')[0] != vid): continue
    print(image)
    full_path = os.path.join(unseens,image)

    im = cv2.imread(full_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    target_size = (hidden_size,432)
    temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
    temp_frame = temp_frame.reshape(1,432,768,3)
    print(temp_frame.shape)
    embd = model.predict(np.array(temp_frame),batch_size=1,verbose=1)['pre_logits']

    # input(embd.shape)
    results = embeddings.query(
        query_embeddings = embd[0][0][0],
        n_results = top_n_closest
    )

    for k in results.keys(): 
        print(f'{k}: {results[k]}')
        print()
    
    pass1_class = determine_class_pass1(results['ids'][0],results['metadatas'][0],results['distances'][0])

    if(pass1_class == 'left'):
        # predicted_lefts.append(embd[0][0][0])
        # predicted_l_ids.append(image)
        temp.append(embd[0][0][0])
        cur_fnames.append(image)
        directions.append({'label':'left','video':vid})
    elif(pass1_class == 'right'):
        # predicted_rights.append(embd[0][0][0])
        # predicted_r_ids.append(image)
        temp.append(embd[0][0][0])
        cur_fnames.append(image)
        directions.append({'label':'right','video':vid})
    elif(pass1_class == 'none'):
        # predicted_nones.append(embd[0][0][0])
        # predicted_n_ids.append(image)
        temp.append(embd[0][0][0])
        cur_fnames.append(image)
        directions.append({'label':'none','video':vid})
    elif(pass1_class == 'pass2'):
        pass2s.append(embd[0][0][0])
        pass2_ids.append(image)
print(len(pass2s))
input('stop')
embeddings.upsert(
            embeddings=temp,
            ids=cur_fnames,
            metadatas = directions
        )

temp = []
cur_fnames = []
directions = []

for i in range(len(pass2s)):
    embd = pass2s[i]
    results = embeddings.query(
        query_embeddings = embd,
        n_results = top_n_closest
    )
    pass2_class = determine_class_pass2(results['ids'][0],results['metadatas'][0],results['distances'][0])
    if(pass2_class == 'left'):
        # predicted_lefts.append(embd[0][0][0])
        # predicted_l_ids.append(image)
        # temp.append(embd)
        cur_fnames.append(pass2_ids[i])
        directions.append({'label':'left','video':vid})
    elif(pass2_class == 'right'):
        # predicted_rights.append(embd[0][0][0])
        # predicted_r_ids.append(image)
        # temp.append(embd)
        cur_fnames.append(pass2_ids[i])
        directions.append({'label':'right','video':vid})
    elif(pass2_class == 'none'):
        # predicted_nones.append(embd[0][0][0])
        # predicted_n_ids.append(image)
        # temp.append(embd)
        cur_fnames.append(pass2_ids[i])
        directions.append({'label':'none','video':vid})

embeddings.upsert(
            embeddings=pass2s,
            ids=cur_fnames,
            metadatas = directions
        )
# print(len(predicted_lefts))
# print(len(predicted_rights))
# print(len(predicted_nones))

# input('stop')
# full_path = '/home/vasantgc/venv/nba_proj/data/unseen_test_images/ims/vid4_frame_5322.jpg'
# im = cv2.imread(full_path)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# target_size = (hidden_size,432)
# temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
# temp_frame = temp_frame.reshape(1,432,768,3)
# print(temp_frame.shape)
# embd = model.predict(np.array(temp_frame),batch_size=1,verbose=1)['pre_logits']

# # input(embd.shape)
# results = embeddings.query(
#     query_embeddings = embd[0][0][0],
#     n_results = 20
# )

# for k in results.keys(): 
#     print(f'{k}: {results[k]}')
#     print()




# print(results)
# print(embd)
# embeddings = [
#     [0.12312,0.74801892849,0.4718234781],
#     [0.659128242,0.7583183479283,0.74921384027],
#     [0.571940834820,0.735948247980,0.142780539123824]
# ]

# ids=['c1','c2','c3']

# metadatas = [
#     {'label': 'vec1'},
#     {'label': 'vec2'},
#     {'label': 'vec3'}
# ]

# collection.add(
#     embeddings=embeddings,
#     ids=ids,
#     metadatas=metadatas
# )

# results = collection.query(
#     query_embeddings=[[0.2, 0.25, 0.35]],
#     n_results=2
# )

# print(results)

# load it back: 
# client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_store"
# ))