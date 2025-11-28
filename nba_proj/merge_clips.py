import os
import shutil

# hardcoded to vid2-------------------
def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def extract_class(clip_name):
    return clip_name.split('_')[3]

def extract_frame_number(frame_name):
    return int(frame_name.split('.')[0].split('_')[2])

def copy_clip(start_frame, end_frame, label):
    cur_clip_count = len(os.listdir(f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}')) + 1
    start_num = extract_frame_number(start_frame)
    end_num = extract_frame_number(end_frame)

    if(not os.path.exists(f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}/{cur_vid}_clip_{cur_clip_count}_{label}')):
        os.makedirs(f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}/{cur_vid}_clip_{cur_clip_count}_{label}')
    # src_dir = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_hmm_final_{cur_vid}'
    src_dir = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/ims'
    dest_dir = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}/{cur_vid}_clip_{cur_clip_count}_{label}'

    for i in range(start_num, end_num+1):
        src_file = os.path.join(src_dir,f'{cur_vid}_frame_{i}.jpg')
        dest_file = os.path.join(dest_dir,f'{cur_vid}_frame_{i}.jpg')
        if(os.path.exists(src_file)):
            shutil.copy(src_file,dest_file)

def delete_non_merged_clip(clip_num,label):
    shutil.rmtree(f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}/{cur_vid}_clip_{clip_num}_{label}')

cur_vid = 'vid4'

src_dir = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_hmm_final_{cur_vid}'
dest_dir = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}'

clips = os.listdir(src_dir)
clips = sorted(clips, key=comparator)

if(os.path.exists(dest_dir)):
    shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    
prev_type = extract_class(clips[0])
interval = []
for clip in clips: 
    cur_type = extract_class(clip)
    # no need to add none clips
    if(cur_type == 'none'):
        continue
    print(cur_type)
    frames = os.listdir(os.path.join(src_dir,clip))
    frames = sorted(frames, key = comparator)

    min_frame = min(frames)
    max_frame = max(frames)

    min_frame_num = extract_frame_number(min_frame)
    max_frame_num = extract_frame_number(max_frame)

    # skip this for vid 4
    # if(max_frame_num-min_frame_num < 200):
    #     continue
    if(interval == []):
        interval.append(extract_frame_number(min_frame))
        interval.append(extract_frame_number(max_frame))
        copy_clip(min_frame,max_frame,cur_type)
        # input('stop')
    else: 
        if(cur_type == prev_type):
            print('possibly need to actually merge')
            print(min_frame)
            print(max_frame)
            print('prev interval: ')
            print(interval)
            # min_frame_num = extract_frame_number(min_frame)
            # max_frame_num = extract_frame_number(max_frame)
            # if cur start is in prev interval, merge
            # if diff between end of last interval and start of cur interval is less than 30, merge (its prolly close enough)
            if(interval[1] > min_frame_num or (min_frame_num - interval[1] <=30)): 
                print(' ------ need to actually merge --------')
                print('previous interval: ')
                print(interval)
                print('current interval: ')
                print(f'[{min_frame_num},{max_frame_num}]')
                print(f'{cur_vid}_frame_{interval[0]}.jpg')
                # input('stop pt 1')
                delete_non_merged_clip(len(os.listdir(f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{cur_vid}')),cur_type)
                # input('stop pt 2')
                copy_clip(f'{cur_vid}_frame_{interval[0]}.jpg',max_frame,cur_type)
                # input('stop pt3')
            else: 
                copy_clip(min_frame,max_frame,cur_type)
            # input('stop at need to merge')
        else:
            copy_clip(min_frame,max_frame,cur_type)
            print('move on')
        print()
        print(interval)
        print(prev_type)
        print()
        interval[0] = extract_frame_number(min_frame)
        interval[1] = extract_frame_number(max_frame)
        prev_type = cur_type
    print('temp break')
    # print(min_frame)
    # print(max_frame)
    # input(frames)
print(os.listdir(dest_dir))

clips = '/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_vid2'
for c in os.listdir(clips):
    print(len(os.listdir(os.path.join(clips,c))))