# load the 3 embeddings clusters
# average them
# subtract one by 10, add one by 10, and leave one neutral?

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from joblib import dump
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
from tensorflow.keras.backend import sigmoid
from sklearn.utils.class_weight import compute_class_weight

cur_vid = 'vid2'
left_data = np.load(f'data/embeddings/{cur_vid}_left_embeddings.npz')
right_data = np.load(f'data/embeddings/{cur_vid}_right_embeddings.npz')
none_data = np.load(f'data/embeddings/{cur_vid}_none_embeddings.npz')

tl = left_data['embeddings']
tr = right_data['embeddings']
tn = none_data['embeddings']

tl = tl.reshape(len(tl),768)
tr = tr.reshape(len(tr),768)
tn = tn.reshape(len(tn),768)

tl_mean = np.mean(left_data['embeddings'],axis=0)[0]
tr_mean = np.mean(right_data['embeddings'],axis=0)[0]
tn_mean = np.mean(none_data['embeddings'],axis=0)[0]

# euclidian distances (very good)
print("Left vs Right:", euclidean(tl_mean, tr_mean))
print("Left vs None:", euclidean(tl_mean, tn_mean))
print("Right vs None:", euclidean(tr_mean, tn_mean))
# Left vs Right: 3.861595392227173
# Left vs None: 5.2430291175842285
# Right vs None: 4.360372066497803

# # cosine distances (very bad)
# print("Left vs Right:", cosine(tl_mean, tr_mean)) 
# print("Left vs None:", cosine(tl_mean, tn_mean))
# print("Right vs None:", cosine(tr_mean, tn_mean))
# # Left vs Right: 0.010567313435242087
# # Left vs None: 0.01952976033597853
# # Right vs None: 0.013695738828088944

print(tl.shape)
print(tr.shape)
print(tn.shape) 
none_indices = np.random.choice(len(tn), size = int(np.mean([len(tl),len(tr)])), replace=False)
tn = tn[none_indices]
# input(none_indices)
# input(tn.shape)
# tn = np.random.choice(tn, size = (np.mean([len(tl),len(tr)])), replace=False)
# input(tn.shape)
yl = [0]*len(tl)
yr = [1]*len(tr)
yn = [2]*len(tn)
combined = np.append(tl, tr,axis=0)
cy = np.append(yl,yr)
combined = np.append(combined, tn,axis=0)
cy = np.append(cy, yn)
# input(combined.shape)
# input(cy.shape)
custom_centroids = np.array([
    tl_mean,
    tr_mean,
    tn_mean
]) 

kmeans = KMeans(n_clusters = 3, 
                init = custom_centroids,
                n_init = 1,
                random_state = 0)
kmeans.fit(combined)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

preds = kmeans.predict(combined)
print(preds.shape)
print(cy.shape)
print(preds == cy)
print(np.sum(preds == cy))

# input(cy[len(tl)+1+len(tr) : len(tl) + len(tr)+len(tn)])
print(f'Class 0 (left side): {np.sum(preds[0:len(tl)] == cy[0:len(tl)])} / {len(tl)}')
print(f'Class 1 (right side): {np.sum(preds[len(tl)+1 : len(tl) + len(tr)] == cy[len(tl)+1 : len(tl) + len(tr)])} / {len(tr)}')
print(f'Class 2 (none): {np.sum(preds[len(tl)+1+len(tr) : len(tl) + len(tr)+len(tn)] == cy[len(tl)+1+len(tr) : len(tl) + len(tr)+len(tn)])} / {len(tn)}')

splitted = train_test_split(combined,cy,train_size=0.95,random_state=0,stratify=cy)
X_train = np.array(splitted[0])
X_test = np.array(splitted[1])
y_train = np.array(splitted[2])
y_test = np.array(splitted[3])


# input(np.unique(y_train,return_counts = True))

# class_weights = compute_class_weight(
#     class_weight = 'balanced',
#     classes = np.unique(y_train),
#     y = y_train
# )

# class_weight_dict = dict(enumerate(class_weights))
# input(class_weight_dict)

class_weight_dict = {0:1.75, 1:1.5, 2:5} # 0:1.2, 1:0.85, 2:1.95
acc_y_train = []
acc_y_test = []

for target in y_train:
    temp = [0.0,0.0,0.0]
    temp[target] = 1.0
    acc_y_train.append(temp)

for target in y_test: 
    temp = [0.0,0.0,0.0]
    temp[target] = 1.0
    acc_y_test.append(temp)

y_train = np.array(acc_y_train)
y_test = np.array(acc_y_test)

# input(X_train.shape)
optimizer = tensorflow.keras.optimizers.Adam(0.0001) #0.00005

input('stop before model creation/training')
# need more none frames
model = Sequential()

# model.add(Input(shape=(768,)))
model.add((Dense(512,activation='relu'))) #512
model.add(Dense(128,activation='relu')) #128
model.add(Dense(3,activation='softmax'))

# input(y_train.shape)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['mse','mae','acc'])
history = model.fit(X_train, 
                    y_train, 
                    epochs=25, 
                    verbose=1, 
                    # class_weight = class_weight_dict,
                    validation_data = (X_test, y_test) 
                    )

model.save(f"{cur_vid}_side_nn.keras")