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
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,6,7"
from joblib import load
import hmm

cur_vid = 'vid5'

# search for CONFIDENT UPSERTS to uncomment those upserts
client = PersistentClient(path="./chroma_store")
embeddings = client.get_or_create_collection(name=f"{cur_vid}_p32_embeddings",metadata={"hnsw:space": "l2"})

layers = tf_keras.layers

# clustering = load_model(f"{cur_vid}_side_nn.keras")

left_path = 'data/unseen_test_images/left'
right_path = 'data/unseen_test_images/right'
none_path = 'data/unseen_test_images/none'
images_folder = f'data/unseen_test_images/ims_{cur_vid}'

test_ims = os.listdir(images_folder)
# input(test_ims)
# print(test_ims)

hidden_size = 768 #768
model = vit.VisionTransformer(
    # image_size = 224,
    input_specs=layers.InputSpec(shape=[None,432,768,3]),  #432,768   648,1152   504,896
    patch_size=32, #16 128 had more variance 64
    num_layers=12, #12  14
    num_heads=12, #12  14
    hidden_size=hidden_size,
    mlp_dim=3072 #3072  3584
)
# 500 was a bit too much, seemed to classify a lot of things as none
top_n_closest = 50
model.load_weights('vit_random_weights.h5')

# temp_hmm = open('temp_file_for_hmm.txt','w')
# this should be 901 (30 seconds of footage is giving around 2 full possessions, which makes sense)
# IGNORE THE SHIFTING THING FOR NOW
# 10000 worked well
hmm_matrix = hmm.hmm(20001) # if you want x frames in the window, do x+1 as the parameter cause we don't use index 0
# hmm_matrix.add_col_to_lattic()

def store_clip(starting_frame,ending_frame,dir): 
    clips_dir = f'data/unseen_test_images/clips_hmm_smooth_{cur_vid}'
    num_clips = len(os.listdir(clips_dir))
    src_path = images_folder
    dest_path = f'{clips_dir}/{starting_frame.split("_")[0]}_clip_{num_clips+1}_{dir}'
    os.mkdir(dest_path)


    start_num = int(starting_frame.split('_')[2].split('.')[0])
    end_num = int(ending_frame.split('_')[2].split('.')[0])

    for i in range(start_num, end_num+1):
        im_name = f'{starting_frame.split("_")[0]}_{starting_frame.split("_")[1]}_{i}.jpg'
        src_file = os.path.join(src_path,im_name)
        dest_file = os.path.join(dest_path,im_name)
        if((not os.path.exists(src_file))):
            continue
        shutil.copy(src_file,dest_file)
    print(num_clips+1)

def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

def temp_smax(x, temperature=1.0):
    x = np.array(x, dtype=np.float64)
    return softmax(x / temperature)

def calculate_context_probs(left_part, right_part):
    counts = {'left':0, 'right': 0, 'none': 0}
    probs = {'left': 0.0, 'right':0.0, 'none':0.0}

    for arr in left_part: 
        counts[arr[0]] += 1
        probs[arr[0]] += arr[1]
    ps = list(probs.values())
    # ps = [probs['left']/counts['left'] if counts['left'] != 0 else 0.0,
    #         probs['right']/counts['right'] if counts['right'] != 0 else 0.0,
    #         probs['none']/counts['none'] if counts['none'] != 0 else 0.0]
    print(ps)
    print(counts)
    final_probs = temp_smax(ps,1.0)
    print('left side')
    print(final_probs)
    counts = {'left':0, 'right': 0, 'none': 0}
    probs = {'left': 0.0, 'right':0.0, 'none':0.0}
    for arr in right_part: 
        counts[arr[0]] += 1
        probs[arr[0]] += arr[1]
    ps = list(probs.values())
    # ps = [probs['left']/counts['left'] if counts['left'] != 0 else 0.0,
    #         probs['right']/counts['right'] if counts['right'] != 0 else 0.0,
    #         probs['none']/counts['none'] if counts['none'] != 0 else 0.0]

    final_probs = temp_smax(ps,10.0)
    print('counts then right side probs')
    print(counts)
    print()
    # input(final_probs)
    print(final_probs)

def generate_clip_intervals(decoded_sequence,frame_names):
        intervals = {}

        intervals = {'left':[],'right':[],'none':[]}

        start_index = 0
        end_index = 0

        cur_type = decoded_sequence[0]
        streak_length = 0
        # TODO: maintain a streak type thing again 
        # like to make it stop adding bs "clips", just make sure that the clip you get
        # actually has more than like 75 frames (only real possessions/blocks seem to have more than 75 continuous ones)
        # obviously 75 is arbitrary, but just large enough to separate continuous possession blocks from noisy frames
        for i in range(len(decoded_sequence)):
            # print(cur_type)
            if(cur_type == decoded_sequence[i]):
                end_index = i
                streak_length += 1
            else:
                if(streak_length > 100):
                    if(cur_type == 'left' or cur_type == 'right'):
                        e = frame_names[end_index]
                        extended_frame = e.split('_')[0]+'_'+e.split('_')[1]
                        new_num = int(e.split('_')[2].split('.')[0]) + 100 # add 100 more frames cause it seems to cut short
                        extended_frame = extended_frame + '_'+str(new_num)+'.jpg'

                        s = frame_names[start_index]
                        prev_frame = s.split('_')[0]+'_'+s.split('_')[1]
                        new_num = int(s.split('_')[2].split('.')[0]) - 100 # add 100 more frames cause it seems to cut short
                        prev_frame = prev_frame + '_'+str(new_num)+'.jpg'


                        # input(s)
                        # input(extended_frame)
                        store_clip(prev_frame,extended_frame,cur_type) # was frame_names[end_index]
                        # intervals[cur_type].append([start_index,end_index])
                streak_length = 0
                start_index = i
                end_index = i
                cur_type = decoded_sequence[i]

        store_clip(frame_names[start_index],frame_names[end_index],cur_type)

def determine_class(ids, metadatas, distances, add_first):
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

    raw_left_percent = num_left / len(ids)
    raw_right_percent = num_right / len(ids)
    raw_none_percent = num_none/ len(ids)

    # nums = [num_left, num_right, num_none]
    # a = np.array(nums).argmax()

    print('inside determine class')
    # input(metadatas)

    probs = {'left_sides':[],'right_sides':[],'none_sides':[]}
    for prob in metadatas: 
        probs['left_sides'].append(prob['left_prob'])
        probs['right_sides'].append(prob['right_prob'])
        probs['none_sides'].append(prob['none_prob'])
    # print(f'left prob: {np.mean(np.array(probs["left_sides"]))}') 
    # print(f'right prob: {np.mean(np.array(probs["right_sides"]))}')
    # print(f'none prob: {np.mean(np.array(probs["none_sides"]))}')
    lmean = np.mean(np.array(probs["left_sides"]))
    rmean = np.mean(np.array(probs["right_sides"]))
    nmean = np.mean(np.array(probs["none_sides"]))

    # print(raw_left_percent)
    # print(raw_right_percent)
    # print(raw_none_percent)
    # input('stop')
    # an average of 0.80 or better can prolly just be written back to the db
    # it looks like perfects have at least a 0.85 average probability, but the lowest perfect i saw was 0.83
    # but maybe its better to add things with a prob >= .90 because thats pretty much guaranteed correct
    
    # input(probs)
    print(f'left prob: {(lmean + raw_left_percent)/2}') 
    print(f'right prob: {(rmean + raw_right_percent)/2}')
    print(f'none prob: {(nmean + raw_none_percent)/2}')
    nums = [(lmean + raw_left_percent)/2, (rmean + raw_right_percent)/2, (nmean + raw_none_percent)/2]
    # input(nums)
    # temp_hmm.write(f"{str({'left':lmean,'right':rmean,'none':nmean})}\n") # CTRLF confidence dicts generated here
    if(add_first == True):
        hmm_matrix.add_first({'left':lmean,'right':rmean,'none':nmean})
    else: 
        hmm_matrix.add_col_to_lattice({'left':lmean,'right':rmean,'none':nmean})
    a = np.array(nums).argmax()

# embeddings.upsert(
#             embeddings=temp,
#             ids=cur_fnames,
#             metadatas = directions
#         )

# temp.append(embd[0][0][0])
#         cur_fnames.append(image)
#         directions.append({'label':'left',
#                             'video':vid,
#                             'left_prob':pass1_class['probs'][0],
#                             'right_prob':pass1_class['probs'][1],
#                             'none_prob':pass1_class['probs'][2]})

    # was 0.85
    confidence_threshold = 0.7 #(lowered it cause we're looking at 100 closest now)
    if(a == 0):
        if(lmean >= confidence_threshold):
            # if all 100 of the closest agree, then we can say with pretty much complete confidence
            # that its correct; could possibly change this later to be like anything over 95
            # top_n_closest is just that 100 (but it needs to be larger because if you ave like 180k frames, you just need more)
            if(num_left == top_n_closest): 
                return ['left',0.999998,{'label':'left',
                    'video':cur_vid,
                    'left_prob':0.999998,
                    'right_prob':0.000001,
                    'none_prob':0.000001}]
            else: 
                return ['left',lmean,{'label':'left',
                    'video':cur_vid,
                    'left_prob':lmean,
                    'right_prob':rmean,
                    'none_prob':nmean}]
        return ['left',lmean]
    elif(a == 1):
        if(rmean >= confidence_threshold):
            if(num_right == top_n_closest):
                return ['right',0.999998,{'label':'right',
                    'video':cur_vid,
                    'left_prob':0.000001,
                    'right_prob':0.999998,
                    'none_prob':0.000001}]
            else:
                return ['right',rmean,{'label':'right',
                        'video':cur_vid,
                        'left_prob':lmean,
                        'right_prob':rmean,
                        'none_prob':nmean}]
        return ['right',rmean]
    else:
        if(nmean >= confidence_threshold):
            if(num_none == top_n_closest):
                return ['none',0.999998,{'label':'none',
                    'video':cur_vid,
                    'left_prob':0.000001,
                    'right_prob':0.000001,
                    'none_prob':0.999998}]
            else:
                return ['none',nmean,{'label':'none',
                    'video':cur_vid,
                    'left_prob':lmean,
                    'right_prob':rmean,
                    'none_prob':nmean}]
        return ['none',nmean]
    
    

test_ims = sorted(test_ims, key = comparator)

clip_intervals = open('clip_intervals.csv','w')
clip_intervals.write('start_frame,end_frame\n')

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

# also, should prolly make this sliding_window a size of 75 isntead of 50 so that theres the same number 
# on both sides (37 on each side)
sliding_window = [] # TODO: should prolly be an actual queue, but this should do for now
starting_frame = ''
ending_frame = ''
window_cap = 75

clips = []
# variables to keep track of how long the current streak is
# the confidence needs to be at least 0.7 to count as part of a streak (scratched this, but possibly implement later)

streaks = {'left': 0, 'right': 0, 'none': 0}


# input(np.array(test_ims))

# need to explicitly store the first and second frame classes 
# decode sequence returns a sequence where the first 2 are -1s, (because its dp the first element is -1 and the second element doesnt have a backpointer)
# for now, just ignore the shifting window thing
# we need to get the clips isolated first
frame_names = []
fully_decoded_sequence = []

first_frame = ''
second_frame = ''
add_first = True

confident_embeds_batch = []
confident_ids_batch = []
confident_metadatas_batch = []

# another thing we should do to speed the process up is just store the test embeddings somewhere as well 
# the write if 80% confident is also taking a while (this is whats taking the most time i think)
# smoothing works well; do windows of 10k instead of 1k
for fname in test_ims:
    # fname = 'frame_' + str(i) + '.jpg'
    # if(cur_vid != fname.)
    # input(fname.split('_')[0])
    # input(cur_vid != fname.split('_')[0])
    
    # for vid2, bounds are 120k to 130k
    # once you do the extended clips, maybe go through each one and do one final check to remove any obviously none frames
    if(cur_vid != fname.split('_')[0]):
        continue
    if(int(fname.split('_')[2].split('.')[0]) <80000) : # was 11000 first; was 18060 ; 109624
        continue
    if(int(fname.split('_')[2].split('.')[0]) == 90000): # was 11000
        break
    print(fname)
    frame_names.append(fname)

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

    if(add_first == True):
        side = determine_class(results['ids'][0],results['metadatas'][0],results['distances'][0],True)
        add_first = False
    else:
        side = determine_class(results['ids'][0],results['metadatas'][0],results['distances'][0],False)
    
    if(side[0] == 'left'):
        cv2.imwrite(f"{left_path}/left_{fname}", im)
        if(len(side) == 3):
            confident_embeds_batch.append(cur_embedding)
            confident_ids_batch.append(fname)
            confident_metadatas_batch.append(side[2])
        # CONFIDENT UPSERTS
        # if(len(side) == 3):
        #     embeddings.upsert(
        #         embeddings = cur_embedding,
        #         ids=fname,
        #         metadatas = side[2]
        #     )
    elif(side[0] == 'right'):
        cv2.imwrite(f"{right_path}/right_{fname}", im)
        if(len(side) == 3):
            confident_embeds_batch.append(cur_embedding)
            confident_ids_batch.append(fname)
            confident_metadatas_batch.append(side[2])
        # CONFIDENT UPSERTS
        # if(len(side) == 3):
        #     embeddings.upsert(
        #         embeddings = cur_embedding,
        #         ids=fname,
        #         metadatas = side[2]
        #     )
    elif(side[0] == 'none'):
        cv2.imwrite(f"{none_path}/none_{fname}", im)
        if(len(side) == 3):
            confident_embeds_batch.append(cur_embedding)
            confident_ids_batch.append(fname)
            confident_metadatas_batch.append(side[2])
        # CONFIDENT UPSERTS
        # if(len(side) == 3):
        #     embeddings.upsert(
        #         embeddings = cur_embedding,
        #         ids=fname,
        #         metadatas = side[2]
        #     )
    
    # 16 was generally the best for everything, but the kyrie video gets everything 
    # but the actual shot either going in or missing (it classifies that as none for some reason)
    # this could be because of how we manually labelled the boundaries
    # just manually add like 50-100 frames to the end of each of the left and right possessions or smthn
    # looks like it consistently misses around 100 ish frames; so just add 100 more frames after every left right frame
    if(len(confident_embeds_batch) == 16): # was 128; 32 worked way better, still a few mishaps
        # CONFIDENT UPSERTS
        print('--------------------upserting confident batch--------------------')
        confident_embeds_batch = np.array(confident_embeds_batch).reshape(len(confident_embeds_batch),len(confident_embeds_batch[0][0]))
        # input((confident_embeds_batch))
        embeddings.upsert(
            embeddings = np.array(confident_embeds_batch),
            ids = confident_ids_batch,
            metadatas = confident_metadatas_batch
        )
        confident_embeds_batch = []
        confident_ids_batch = []
        confident_metadatas_batch = []

    if(first_frame == ''): 
        first_frame = side[0]
    elif(second_frame == ''):
        second_frame = side[0]
    # if(len(sliding_window)<30):
    #     sliding_window.append(side)
    # else: 
    #     input(sliding_window)

    print(side)

fully_decoded_sequence = hmm_matrix.decode_sequence()
fully_decoded_sequence[0] = first_frame
fully_decoded_sequence[1] = second_frame

print(classes_confidences)

generate_clip_intervals(fully_decoded_sequence,frame_names)


# print(hmm_matrix.decode_sequence())

