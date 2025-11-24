import os
from chromadb import PersistentClient
from official.vision.modeling.backbones import vit
import cv2
import tensorflow as tf, tf_keras
import numpy as np
import hmm
import shutil

layers = tf_keras.layers
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,7" # these have the most memory
# so run the hmm on this between each possession and see if it actually encodes it properly
# if it does split it properly, the encoding should match
# if it didnt split properly, 
# then we should see a different encoding (like the hmm outputs 'none' 'none' but it was labelled as 'left' 'left)


client = PersistentClient(path="./chroma_store")
cur_vid = 'vid4'
embeddings = client.get_or_create_collection(name=f"{cur_vid}_p32_embeddings",metadata={"hnsw:space": "l2"})


hmm_matrix = hmm.hmm(15000)

def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def determine_class(ids, metadatas, distances, add_first):
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
    
    # input(probs)
    # print(f'left prob: {(lmean + raw_left_percent)/2}') 
    # print(f'right prob: {(rmean + raw_right_percent)/2}')
    # print(f'none prob: {(nmean + raw_none_percent)/2}')
    # nums = [(lmean + raw_left_percent)/2, (rmean + raw_right_percent)/2, (nmean + raw_none_percent)/2]

    nums = [(lmean + raw_left_percent)/2, (rmean + raw_right_percent)/2, (nmean + raw_none_percent)/2]
    # input(nums)
    # temp_hmm.write(f"{str({'left':lmean,'right':rmean,'none':nmean})}\n") # CTRLF confidence dicts generated here
    if(add_first == True):
        hmm_matrix.add_first({'left':lmean,'right':rmean,'none':nmean})
    else: 
        hmm_matrix.add_col_to_lattice({'left':lmean,'right':rmean,'none':nmean})
    a = np.array(nums).argmax()

    if(a == 0):
        return ['left',lmean,{'label':'left',
            'video':cur_vid,
            'left_prob':lmean,
            'right_prob':rmean,
            'none_prob':nmean}]
    elif(a == 1):
        return ['right',rmean,{'label':'right',
                'video':cur_vid,
                'left_prob':lmean,
                'right_prob':rmean,
                'none_prob':nmean}]
    else:
        return ['none',nmean,{'label':'none',
            'video':cur_vid,
            'left_prob':lmean,
            'right_prob':rmean,
            'none_prob':nmean}]
        
    # input(nums)
def copy_im(src_file, dest_file):
    shutil.copy(src_file,dest_file)
# this is how you copy things into a new file
# shutil.copy(src_file,dest_file)

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
top_n_closest = 5
model.load_weights('vit_random_weights.h5')

main_dir = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_hmm_smooth_{cur_vid}'
# main_dir = '/home/vasantgc/venv/nba_proj/data/unseen_test_images/padded_v2'
save_to_dir_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_hmm_final_{cur_vid}'

clip_directories = os.listdir(main_dir)
# input(clip_directories)
add_first = True
for clip in clip_directories: 
    clip_path = str(os.path.join(main_dir,clip))
    frames_in_clip = os.listdir(clip_path)
    label = clip.split('_')[-1]
    if(os.path.exists(os.path.join(save_to_dir_root,clip))):
        print(f'path {clip} exists')
        continue
    # input(label)
    frames_in_clip = sorted(frames_in_clip, key = comparator)
    # input(frames_in_clip)
    for fname in frames_in_clip:
        full_path = os.path.join(clip_path,fname)
        # input(full_path)
        im = cv2.imread(full_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        target_size = (hidden_size,432)
        temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)


        aux = []
        aux.append(temp_frame)
        # print(np.array(aux).shape)
        output = model.predict(np.array(aux), batch_size = 1, verbose=1)


        cur_embedding = output['pre_logits']
        cur_embedding = cur_embedding.reshape(1,768)

        # input(cur_embedding)
        results = embeddings.query(
            query_embeddings = cur_embedding,
            n_results = top_n_closest
        )

        print(full_path)
        if(add_first == True):
            side = determine_class(results['ids'][0],results['metadatas'][0],results['distances'][0],True)
            add_first = False
        else:
            side = determine_class(results['ids'][0],results['metadatas'][0],results['distances'][0],False)

        # print('side')
    decoded = hmm_matrix.decode_sequence()
    # print(decoded)
    print(len(frames_in_clip))
    print(len(decoded))

    zipped = list(zip(frames_in_clip,decoded))

    for combo in zipped: 
        if(combo[1] == label):
            # print(combo)
            src = os.path.join(clip_path,combo[0])
            dst = os.path.join(save_to_dir_root,clip)
            if(not os.path.exists(dst)):
                os.mkdir(dst)
            copy_im(src,os.path.join(dst,combo[0]))
    hmm_matrix = hmm.hmm(15000)
    # input(zipped)
    # clip path is the source 

    # print(hmm_matrix.decode_sequence())
    # input('stop')
        # input(side)
        # determine_class(results['ids'][0],results['metadatas'][0],results['distances'][0])

        # input(frames_in_clip)

# print(clip_directories)