# load the 3 embeddings clusters
# average them
# subtract one by 10, add one by 10, and leave one neutral?

import numpy as np
from scipy.spatial.distance import cosine, euclidean

left_data = np.load('left_embeddings.npz')
right_data = np.load('right_embeddings.npz')
none_data = np.load('none_embeddings.npz')

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

# cosine distances (very bad)
print("Left vs Right:", cosine(tl_mean, tr_mean)) 
print("Left vs None:", cosine(tl_mean, tn_mean))
print("Right vs None:", cosine(tr_mean, tn_mean))
# Left vs Right: 0.010567313435242087
# Left vs None: 0.01952976033597853
# Right vs None: 0.013695738828088944