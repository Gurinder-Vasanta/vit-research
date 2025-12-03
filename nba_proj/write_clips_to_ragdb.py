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
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

hidden_size = 768 #768
layers = tf_keras.layers

model = vit.VisionTransformer(
        # image_size = 224,
        input_specs=layers.InputSpec(shape=[None,432,768,3]),  #432,768   648,1152   504,896
        patch_size=32, #16 128 had more variance 64
        num_layers=12, #12  14
        num_heads=12, #12  14
        hidden_size=hidden_size,
        mlp_dim=3072 #3072  3584
    )

model.load_weights('vit_random_weights.h5')

def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def get_clip_num(name):
    return int(name.split('_')[2])

def get_frame_num(name): 
    return int(name.split('_')[2].split('.')[0])

def get_side(name):
    return name.split('_')[3]

def im_to_array(frame_path): 
    im = cv2.imread(frame_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    target_size = (hidden_size,432)
    temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)
    return temp_frame


def gen_embeddings(aux):
    # im = cv2.imread(frame_path)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # target_size = (hidden_size,432)
    # temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)

    # aux = []
    # aux.append(temp_frame)
    # print(np.array(aux).shape)
    batch_size = 128
    output = model.predict(np.array(aux), batch_size = batch_size, verbose=1)

    cur_embedding = output['pre_logits']
    cur_embedding = cur_embedding.reshape(len(aux),768)
    return cur_embedding


client = PersistentClient(path="./chroma_store")
ragdb = client.get_or_create_collection(name=f"ragdb_p32_embeddings",metadata={"hnsw:space": "l2"})

vids = ['vid2','vid4']
# vids = ['vid5']

# ragdb schema:  
# clipnum
# framenum
# t_norm (frame num (relative to this clip)/num frames in clip)
# side

# clip_start_range = 59
# clip_end_range = 70

batch_cap = 128

for vid in vids: 
    all_clips_path = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}'
    clips = os.listdir(all_clips_path)
    clips = sorted(clips,key=comparator)

    for clip in clips:
        print('cur clip: ')
        print(clip)
        # if(int(clip.split('_')[2]) < clip_start_range):
        #     continue
        # cur_count = 0
        # input(clip)

        pre_embeddings = []
        # t_norm = []
        ids = []
        metadatas = []

        vid_clip = get_clip_num(clip)

        cur_clip_path = os.path.join(all_clips_path,clip)
        frames = os.listdir(cur_clip_path)
        frames = sorted(frames,key=comparator)

        f_counter = 1
        cur_counter = 0
        num_processed_frames = 0
        # frames = frames[0:5]
        # input(frames)
        for f in frames: 
            frame_num = get_frame_num(f)

            pth = os.path.join(cur_clip_path,f)
            temp = im_to_array(pth)
            pre_embeddings.append(temp)
            # t_norm.append()
            ids.append(f)
            metadatas.append({'side': get_side(clip),
                              't_norm':f_counter/len(frames),
                              'clip_num': get_clip_num(clip),
                              'vid_num':int(f.split('_')[0][3::])})

            f_counter += 1
            cur_counter += 1
            num_processed_frames += 1

            if(cur_counter == batch_cap):
                # input(pre_embeddings)
                embeddings = gen_embeddings(pre_embeddings)
                print(f'        clip: {clip}')
                print(f'        start frame: {f}')
                print(f'        processed {num_processed_frames} / {len(frames)}')
                # input(embeddings)
                # print(ids)
                # print(metadatas)
                # input('stop')

                # input(np.array(embeddings).shape)

                ragdb.upsert(
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )

                embeddings = []
                pre_embeddings = []
                ids = []
                metadatas = []
                cur_counter = 0

                # input('stop')
            # input(temp)

        # print(embeddings)
        # print(ids)
        # print(metadatas)
        if((len(frames) - num_processed_frames) < batch_cap and (len(pre_embeddings) != 0)):
            embeddings = gen_embeddings(pre_embeddings)
            print(f'        clip: {clip}')
            print(f'        start frame: {f}')
            print(f'        processed {num_processed_frames} / {len(frames)}')
            # input(embeddings)
            ragdb.upsert(
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas
                    )

            embeddings = []
            ids = []
            metadatas = []
            cur_counter = 0
        num_processed_frames = 0
        # input(frames)
    # input(clips)

