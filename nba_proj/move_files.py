import shutil
import os 

src_path = 'data/unseen_test_images/ims'
dest_path = 'data/temp'

for i in range(4156,5501):
    im_name = 'frame_'+str(i)+'.jpg'
    src_file = os.path.join(src_path, im_name)
    dst_file = os.path.join(dest_path, im_name)
    shutil.move(src_file,dst_file)