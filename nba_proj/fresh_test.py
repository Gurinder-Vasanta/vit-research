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

layers = tf_keras.layers

optimizer = tensorflow.keras.optimizers.Adam(0.0001)

clustering = load_model("side_nn.keras")
# clustering = Sequential()

# # model.add(Input(shape=(768,)))
# clustering.add((Dense(250,activation='relu')))
# clustering.add(Dense(128,activation='relu'))
# clustering.add(Dense(3,activation='softmax'))

# clustering.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['mse','mae','acc'])
# # clustering = load('side_clustering.joblib')
# clustering.load_weights("side_nn.weights.h5")
# 'data/temp'

images_folder = 'data/unseen_test_images/ims'
left_path = 'data/unseen_test_images/left'
right_path = 'data/unseen_test_images/right'
none_path = 'data/unseen_test_images/none'

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

model.load_weights('vit_random_weights.h5')

for fname in test_ims:
    # fname = 'frame_' + str(i) + '.jpg'
    print(fname)
    full_path = os.path.join(images_folder,fname)
    im = cv2.imread(full_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    target_size = (hidden_size,432)
    temp_frame = cv2.resize(im,target_size,interpolation=cv2.INTER_AREA)


    aux = []
    aux.append(temp_frame)
    print(np.array(aux).shape)
    output = model.predict(np.array(aux), batch_size = 32, verbose=1)


    cur_embedding = output['pre_logits']
    cur_embedding = cur_embedding.reshape(1,768)
    print(cur_embedding.shape)
    side = clustering.predict(cur_embedding)
    # confidences = clustering.decision_function(cur_embedding)
    print(side)
    # print(confidences)

    pred_side = np.array(side).argmax()
    if(pred_side == 0):
        cv2.imwrite(f"{left_path}/left_{fname}", im)
    elif(pred_side == 1):
        cv2.imwrite(f"{right_path}/right_{fname}", im)
    elif(pred_side == 2):
        cv2.imwrite(f"{none_path}/none_{fname}", im)
    # input('stop')

# print(test_ims[0])