from official.vision.modeling.backbones import vit
import tensorflow as tf
import numpy as np
import tensorflow as tf, tf_keras
import skvideo.io
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    input_specs=layers.InputSpec(shape=[None,432,768,3]),
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
frames = []

max_frames = 10240
train_ind,test_ind = train_test_split([i for i in range(max_frames)],train_size=0.8,random_state=0)

while True:
    if(len(embeddings) == max_frames): break #10240
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
    if(cur_count == batch_cap):
        aux = np.array(aux)
        # input(aux.shape)
    
        output = model.predict(aux, batch_size = 32, verbose=1)
        aux = []
        cur_count = 0
        # print(output)
        print(output['pre_logits'].shape) # generates a 768 length vector for each frame, so the shape is 256,1,1,768
        temp = np.array(output['pre_logits'])
        temp = temp.reshape(batch_cap,1,768)
        for embd in temp:
            embeddings.append(embd)
        print(np.array(embeddings).shape)
    else:
        cur_count += 1
        target_size = (768,432)
        temp_frame = cv2.resize(frame,target_size,interpolation=cv2.INTER_AREA)
        aux.append(temp_frame)




X_train,X_test = train_test_split(embeddings,train_size=0.8,random_state=0)

X_train = np.array(X_train)
X_test = np.array(X_test)
print('--------------')
print(np.array(X_train).shape)
print(np.array(X_test).shape)

X_train = X_train.reshape(X_train.shape[0],768)
X_test = X_test.reshape(X_test.shape[0],768)

kmeans = KMeans(n_clusters = 5, random_state = 0)
kmeans.fit(X_train)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(labels)
print(centroids)

for i in range(10):
    temp = frames[train_ind[i]]
    temp_rgb = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
    plt.imshow(temp_rgb)
    plt.savefig(f'test_images/train_ind_{[train_ind[i]]}_labelled{[labels[0]]}.png')

# temp = frames[train_ind[1]]
# temp_rgb = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
# plt.imshow(temp_rgb)
# plt.savefig(f'train_ind_{[train_ind[1]]}_labelled{[labels[1]]}.png')

# plt.show()

# print(kmeans.predict(X_test))




# K = np.arange(100)+1
# grid = {'n_neighbors':K}

# knnCV = GridSearchCV(knn, param_grid = grid, return_train_score = True)
# knnCV.fit(X_train, y_train)