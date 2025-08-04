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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from joblib import load

cur_vid = 'vid2'
layers = tf_keras.layers

clustering = load_model(f"{cur_vid}_side_nn.keras")

left_path = 'data/unseen_test_images/left'
right_path = 'data/unseen_test_images/right'
none_path = 'data/unseen_test_images/none'
images_folder = 'data/unseen_test_images/ims'

test_ims = os.listdir(images_folder)
# print(test_ims)

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
top_n_closest = 25
model.load_weights('vit_random_weights.h5')

def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def determine_class(ids, metadatas, distances):
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

    # nums = [num_left, num_right, num_none]
    # a = np.array(nums).argmax()

    print('inside determine class')
    # input(metadatas)

    probs = {'left_sides':[],'right_sides':[],'none_sides':[]}
    for prob in metadatas: 
        probs['left_sides'].append(prob['left_prob'])
        probs['right_sides'].append(prob['right_prob'])
        probs['none_sides'].append(prob['none_prob'])
    print(f'left prob: {np.mean(np.array(probs["left_sides"]))}') 
    print(f'right prob: {np.mean(np.array(probs["right_sides"]))}')
    print(f'none prob: {np.mean(np.array(probs["none_sides"]))}')
    lmean = np.mean(np.array(probs["left_sides"]))
    rmean = np.mean(np.array(probs["right_sides"]))
    nmean = np.mean(np.array(probs["none_sides"]))
    # an average of 0.80 or better can prolly just be written back to the db
    # it looks like perfects have at least a 0.85 average probability, but the lowest perfect i saw was 0.83
    
    # input(probs)

    nums = [lmean, rmean, nmean]
    a = np.array(nums).argmax()

    if(a == 0):
        return 'left'
    elif(a == 1):
        return 'right'
    else:
        return 'none'
    
    

test_ims = sorted(test_ims, key = comparator)
client = PersistentClient(path="./chroma_store")

embeddings = client.get_or_create_collection(name=f"{cur_vid}_embeddings",metadata={"hnsw:space": "l2"})

# current_clip = []
# all_clips = []

# current_side_info = {'side': 0,'count':0}

classes_confidences = {}

# this sliding window array would lets us know when a switch happened
# so it would be 30 frames long 
# so theoretically, if there were like 30 lefts, and then like 10 nones started to appear, 
# then we know that the clip ends there 
# so that means you need to track where the start is in a different variable prolly
# for reference, I would just be using the name of the frame itself

sliding_window = []
starting_frame = ''
ending_frame = ''
# input(np.array(test_ims))
for fname in test_ims:
    # fname = 'frame_' + str(i) + '.jpg'
    # if(cur_vid != fname.)
    # input(fname.split('_')[0])
    # input(cur_vid != fname.split('_')[0])
    if(cur_vid != fname.split('_')[0]):
        continue
    if(int(fname.split('_')[2].split('.')[0]) <11000): 
        continue
    print(fname)
    temp_split = fname.split('_')
    # if(temp_split[0] == 'vid1'): continue
    # if(int(temp_split[2].split('.')[0]) < 12000): continue
    full_path = os.path.join(images_folder,fname)
    im = cv2.imread(full_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    target_size = (hidden_size,432)
    temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)


    aux = []
    aux.append(temp_frame)
    # print(np.array(aux).shape)
    output = model.predict(np.array(aux), batch_size = 32, verbose=1)


    cur_embedding = output['pre_logits']
    cur_embedding = cur_embedding.reshape(1,768)

    results = embeddings.query(
        query_embeddings = cur_embedding,
        n_results = top_n_closest
    )

    for k in results.keys(): 
        print(f'{k}: {results[k]}')
        print()
    # input('stop')
    side = determine_class(results['ids'][0],results['metadatas'][0],results['distances'][0])

    if(side == 'left'):
        cv2.imwrite(f"{left_path}/left_{fname}", im)
    elif(side == 'right'):
        cv2.imwrite(f"{right_path}/right_{fname}", im)
    elif(side == 'none'):
        cv2.imwrite(f"{none_path}/none_{fname}", im)

    # if(len(sliding_window)<30):
    #     sliding_window.append(side)
    # else: 
    #     input(sliding_window)
    print(side)
    # print(cur_embedding.shape)
    # side = clustering.predict(cur_embedding)[0]
    # confidences = clustering.decision_function(cur_embedding)

    # classes_confidences[fname] = np.ndarray.tolist(side)
    # i guess 85% confidence is confident enough? this is arbitrary though
    # if(side[0] >= 0.85 and current_side_info['side'] == 0):
    #     current_side_info['side'] = 0
    #     current_side_info['count'] += 1
    #     current_clip.append(fname)
    # print(side)
    # if(int(fname.split('_')[2].split('.')[0]) == 11358):
    #     input(current_clip)
    # input(current_clip)
    # input(side)


    # print(confidences)

    # pred_side = np.array(side).argmax()
    # if(pred_side == 0):
    #     cv2.imwrite(f"{left_path}/left_{fname}", im)
    # elif(pred_side == 1):
    #     cv2.imwrite(f"{right_path}/right_{fname}", im)
    # elif(pred_side == 2):
    #     cv2.imwrite(f"{none_path}/none_{fname}", im)

print(classes_confidences)