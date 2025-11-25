import os
import rag_vit
# import 
import tensorflow as tf, tf_keras
import cv2
import numpy as np


layers = tf_keras.layers

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

path = '/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_vid2_old'
clips = os.listdir(path)

count = 0
for clip in clips: 
    print(clip)
    print(len(os.listdir(os.path.join(path,clip))))
    if(len(os.listdir(os.path.join(path,clip))) >= 200):
        count += 1
print(count)

hidden_size = 768 #768

full_path = '/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_vid2_old/vid2_clip_1_left/vid2_frame_18072.jpg'

im = cv2.imread(full_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

target_size = (hidden_size,432)
temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)

print(temp_frame.shape)
aux = []
aux.append(temp_frame)

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

model = rag_vit.VisionTransformer(
    # image_size = 224,
    input_specs=layers.InputSpec(shape=[None,432,768,3]),  #432,768   648,1152   504,896
    patch_size=32, #16 128 had more variance 64
    num_layers=12, #12  14
    num_heads=12, #12  14
    hidden_size=hidden_size,
    mlp_dim=3072 #3072  3584
)

input('jlkjlkjklj')
output = model.predict(np.array(aux), batch_size = 32, verbose=1)

print(output)

attentions = output['attention_scores']
tokens = output['5']
cur_embedding = output['pre_logits']
cur_embedding = cur_embedding.reshape(1,768)

print(attentions.shape)
print(tokens.shape)
print(cur_embedding.shape)


