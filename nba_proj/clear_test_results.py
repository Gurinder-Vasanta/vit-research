import os

paths = ['data/unseen_test_images/left','data/unseen_test_images/right','data/unseen_test_images/none']
# path = 'data/unseen_test_images/none'
for path in paths:
    files = os.listdir(path)
    for name in files: 
        os.remove(os.path.join(path,name))