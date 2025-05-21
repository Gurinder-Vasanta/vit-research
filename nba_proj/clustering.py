# load the 3 embeddings clusters
# average them
# subtract one by 10, add one by 10, and leave one neutral?

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

left_data = np.load('data/embeddings/left_embeddings.npz')
right_data = np.load('data/embeddings/right_embeddings.npz')
none_data = np.load('data/embeddings/none_embeddings.npz')

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

splitted = train_test_split(combined,cy,train_size=0.8,random_state=0,stratify=cy)
X_train = np.array(splitted[0])
X_test = np.array(splitted[1])
y_train = np.array(splitted[2])
y_test = np.array(splitted[3])

clf = LogisticRegression(max_iter=1000,verbose=1)
clf.fit(X_train, y_train)
 
print(X_test.shape)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
# print(cy[0:len(tl)])
