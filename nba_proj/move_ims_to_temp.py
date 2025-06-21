import os
import shutil
vid = 'vid2'
start = 1
end = 7000

src_path = 'data/unseen_test_images/ims'
dest_path = 'data/temp'

for i in range(start, end+1):
    im_name = vid+'_frame_'+str(i)+'.jpg'
    src_file = os.path.join(src_path, im_name)
    dst_file = os.path.join(dest_path, im_name)
    shutil.move(src_file,dst_file)
