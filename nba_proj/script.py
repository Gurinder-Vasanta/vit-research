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
from sklearn.preprocessing import normalize

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

cap = cv2.VideoCapture(output_path)
batch_cap = 256
cur_count = 0
embeddings = []
aux = []
frames = []

max_frames = 1024 #10240  12288
train_ind,test_ind = train_test_split([i for i in range(max_frames)],train_size=0.8,random_state=0)

global_counter = 0
while True:
    if(len(embeddings) == max_frames): break #10240
    ret, frame = cap.read()
    if not ret: break
    global_counter+=1
    if(global_counter < 514): continue
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
        temp = temp.reshape(batch_cap,1,hidden_size)
        for embd in temp:
            embeddings.append(embd)
        print(np.array(embeddings).shape)
    else:
        cur_count += 1
        target_size = (hidden_size,432) #504
        cv2.imwrite(f"data/temp/frame_{global_counter}.jpg", frame)
        temp_frame = cv2.resize(frame,target_size,interpolation=cv2.INTER_AREA)
        
        aux.append(temp_frame)

# these are labelled frames (put the manually labelled ones in the temp folder)
# 1-420 is left
# 421 to 458 is none
# 458 to 896 is right
# 897 to 953 is none
# 954 to 1303 is right
# 1304 to 1540+ is left

# going to write these to a npz file to load embeddings from them
# the embeddings are going to be stored in the data/left data/none data/right folders
input('stop')
# input(np.array(embeddings).shape)
# for i in range(len(embeddings)):
#     embeddings[i] = embeddings[i] * 1000
    # input(embeddings[i].shape)
X_train,X_test = train_test_split(embeddings,train_size=0.8,random_state=0)

# pick some frames that represents a hardcoded centroid
# do these steps: 
# pick a frame (or a few frames) that represents lhs, rhs, non
# generate embeddings for these which will become the new hardcoded centroids
# do the kmeans on that
X_train = np.array(X_train)
X_test = np.array(X_test)
print('--------------')
print(np.array(X_train).shape)
print(np.array(X_test).shape)

X_train = X_train.reshape(X_train.shape[0],hidden_size)
X_test = X_test.reshape(X_test.shape[0],hidden_size)

X_train = normalize(X_train,norm='l2') #l2 worked somewhat well
X_test = normalize(X_test,norm='l2')

# X_train = X_train * 1000
# X_test = X_test * 1000
for i in range(len(X_train)):
    embeddings[i] = embeddings[i] * 1000

print(X_train)
custom_centroids = np.array([
    X_train[185],
    X_train[6419],
    X_train[4066]
]) #train_ind 185 (for left); 6419 (for right); 4066 (for neither)
kmeans = KMeans(n_clusters = 3, 
                # init = custom_centroids,
                # n_init = 1,
                random_state = 0)
kmeans.fit(X_train)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(labels)
print(centroids)

for cluster_id in range(3):
    for i in np.where(labels == cluster_id)[0][:5]:
        cv2.imwrite(f"test_images/cluster_{cluster_id}_frame_{i}.jpg", frames[i])


# for i in range(30):
#     temp = frames[train_ind[i]]
#     temp_rgb = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
#     plt.imshow(temp_rgb)
#     plt.savefig(f'test_images/train_ind_{[train_ind[i]]}_labelled{[labels[i]]}.png')




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