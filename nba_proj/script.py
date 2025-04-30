from official.vision.modeling.backbones import vit
import tensorflow as tf
import numpy as np
import tensorflow as tf, tf_keras
import skvideo.io
import cv2
layers = tf_keras.layers
# print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
# from tensorflow.python.platform import build_info as tf_build_info
# print(tf_build_info.cuda_version_number)
# print(tf_build_info.cudnn_version_number)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '3' to suppress everything
  
#1080 rows
#1920 cols
#3 channels
inputparameters = {}
outputparameters = {}

output_path = 'data/hoop_ball_finding.mp4'
# reader = skvideo.io.FFmpegReader(output_path,
#                                  inputdict=inputparameters,
#                                  outputdict=outputparameters)

model = vit.VisionTransformer(
    # image_size = 224,
    input_specs=layers.InputSpec(shape=[None,224,224,3]),
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_size=768,
    mlp_dim=3072
)

cap = cv2.VideoCapture(output_path)
batch_cap = 256
cur_count = 0
embeddings = []
aux = []
while True:
    ret, frame = cap.read()
    if not ret: break
    if(cur_count == batch_cap):
        aux = np.array(aux)
        print(aux.shape)
        output = model.predict(aux, batch_size = 32, verbose=1)
        aux = []
        cur_count = 1
        input(output)
    else:
        cur_count += 1
        temp_frame = np.array(frame)
        aux.append(temp_frame)

