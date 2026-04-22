import json
import pprint

import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from loader import preprocess_frame
import pandas as pd
import os

import tensorflow.keras as tf_keras

# -------------------------
# 1. Load labels + build samples
# -------------------------

def comparator(fname):
    splitted = fname.split('_')
    vid_num = int(splitted[0][3::])
    frame_num = int(splitted[2].split('.')[0])
    return (vid_num, frame_num)

def get_fnum(fname): 
    return int(fname.split('_')[2].split('.')[0])

def oversample_chunk_samples(chunk_samples, target="max", seed=1234):
    """
    Oversample by status_id:
      0 = event-none
      1 = event-miss
      2 = event-make

    target:
      - "max": oversample every class up to the largest class count
      - int: oversample every class up to this number times the majority class
    """
    rng = np.random.default_rng(seed)

    by_class = {0: [], 1: [], 2: []}
    for c in chunk_samples:
        by_class[int(c["status_id"])].append(c)

    counts = {k: len(v) for k, v in by_class.items()}
    print("original chunk counts:", counts)

    if target == "max":
        target_count = max(counts.values())
    else:
        target_count = int(target*counts[0])

    out = []
    for cls, items in by_class.items():
        if len(items) == 0:
            continue

        if len(items) >= target_count:
            sampled = items
        else:
            extra_idx = rng.choice(len(items), size=target_count - len(items), replace=True)
            sampled = items + [items[i] for i in extra_idx]

        out.extend(sampled)

    rng.shuffle(out)

    new_counts = {
        0: sum(int(x["status_id"]) == 0 for x in out),
        1: sum(int(x["status_id"]) == 1 for x in out),
        2: sum(int(x["status_id"]) == 2 for x in out),
    }
    print("oversampled chunk counts:", new_counts)

    return out

def load_samples(train_vids, stride = 1, max_clips = 30, start_clip = 0, end_clip=30):
    already_labelled = pd.read_csv("clips_label.csv")
    with open('clip_labelling_template.json','r') as f: 
        data = json.load(f)
    # pprint.pprint(data)
    # print('jfkalsdjlkjsal')
    # input('data')
    samples = []

    print(train_vids)
    for vid in train_vids:
        # clip_root = f"/home/.../clips_finalized_{vid}"

        print(vid)
        # clip_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}'
        clip_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_{vid}_smart'
        clips = sorted(os.listdir(clip_root),key=comparator) 
    
        clips = clips[start_clip:end_clip] #this should change to 50:50+max_clips (because first 50 will get added )
        for clip in clips:
            clip_path = os.path.join(clip_root, clip)
            frames = sorted(os.listdir(clip_path),key=comparator) 
            
            # find label
            label_row = already_labelled[
                already_labelled["clip_path"] == clip_path
            ]
            # print(len(np.array(label_row)[0]))
            if label_row.empty or pd.isna(label_row['label'].iloc[0]):
                clip_label= -1 #-1 means its not labelled, so any chunk with this label is true inference
                # print('???')
                # continue
            else: 
                clip_label = int(label_row["label"].iloc[0])

            num_frames = len(frames)
            stride_counter = 0
            for i, f in enumerate(frames, start=1):
                fpath = os.path.join(clip_path, f)
                # print(f)
                # print(fpath)
                fnum = get_fnum(f)
                # print(fnum)
                # pprint.pprint(data[clip_path])
                makes = data[clip_path]['event_make']
                misses = data[clip_path]['event_miss']
                not_applicable = data[clip_path]['event_none']
                # print(makes)
                # print(misses)
                # print(not_applicable)

                status = ''
                status_id = -1
                for arr in makes: 
                    if(fnum >= arr[0] and fnum <= arr[1]):
                        status = 'event-made'
                        status_id = 2
                for arr in misses: 
                    if(fnum >= arr[0] and fnum <= arr[1]):
                        status = 'event-miss'
                        status_id = 1
                for arr in not_applicable:
                    # print(arr)
                    if(fnum >= arr[0] and fnum <= arr[1]):
                        status = 'event-none'
                        status_id = 0
                # print([status, status_id])
                # input('stop')
                
                temp_tnorm = i / num_frames
                # if(temp_tnorm < 0.7):
                #     continue

                stride_counter += 1

                if(stride_counter == stride):
                    samples.append({
                        "pth": fpath,
                        "side": clip.split("_")[3],
                        "t_norm": i / num_frames,
                        "clip_num": int(clip.split("_")[2]),
                        "vid_num": int(f.split("_")[0][3:]),
                        "label": clip_label,
                        "status": status,
                        "status_id": status_id
                    })

                    stride_counter = 0
    # input(np.array(samples))
    return samples

def chunk_event_label(frame_event_labels, event_threshold = 3):
        """
        Convert frame-level event labels into one chunk-level local label.

        Rule:
        - if make-event frames >= threshold and >= miss-event frames -> make-event
        - elif miss-event frames >= threshold and > make-event frames -> miss-event
        - else -> no-event
        """
        make_count = sum(int(x == 2) for x in frame_event_labels)
        miss_count = sum(int(x == 1) for x in frame_event_labels)

        if make_count >= event_threshold and make_count >= miss_count:
            return 2
        if miss_count >= event_threshold and miss_count > make_count:
            return 1
        return 0

event_lookups = {
    0: 'event-none',
    1: 'event-miss',
    2: 'event-make'
}
def build_chunks(frame_samples, chunk_size=12, chunk_stride=4):
    """
    frame_samples: list of dictionaries (one per frame)
    chunk_size: number of frames per chunk
    chunk_stride: step size between consecutive chunks

    Returns: list of chunk dictionaries
    """

    if chunk_stride <= 0:
        raise ValueError(f"chunk_stride must be positive, got {chunk_stride}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_stride > chunk_size:
        print(f"[warn] chunk_stride ({chunk_stride}) > chunk_size ({chunk_size}); chunks will not overlap")

    # --- Group frames by clip ---
    clips = {}
    for s in frame_samples:
        key = (s["vid_num"], s["clip_num"])
        if key not in clips:
            clips[key] = []
        clips[key].append(s)

    # print(frame_samples[0])
    # input('stop')
    # --- Sort frames within each clip ---
    for key in clips:
        clips[key].sort(key=lambda x: x["t_norm"])

    # --- Build chunks ---
    chunk_samples = []

    for (vid, clip), frames in clips.items():
        total_frames = len(frames)

        if total_frames < chunk_size:
            continue

        label = frames[0]["label"]
        side = frames[0]["side"]

        # overlapping sliding window
        for start in range(0, total_frames - chunk_size + 1, chunk_stride):
            end = start + chunk_size
            sub = frames[start:end]

            frame_paths = [f["pth"] for f in sub]
            stats = [f["status"] for f in sub]
            stat_ids = [f["status_id"] for f in sub]

            # print(stats)
            # print(stat_ids)
            t_vals = [f["t_norm"] for f in sub]
            t_center = float(sum(t_vals) / len(t_vals))
            t_width = float(max(t_vals) - min(t_vals))

            chunk_samples.append({
                "frames": frame_paths,
                "label": label,
                "status": event_lookups[chunk_event_label(stat_ids)],
                "status_id": chunk_event_label(stat_ids),
                "side": side,
                "vid": vid,
                "clip": clip,
                "t_center": t_center,
                "t_width": t_width,
                "start_idx": start,
                "end_idx": end - 1,
            })

    return chunk_samples

def load_and_preprocess_image(path, img_size=(432, 768)):
    """
    path: scalar string tensor
    returns: float32 tensor in range [0,1], shape = (H, W, 3)
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size)
    return img

# def parse_chunk(chunk, img_size=(432,768)):
#     """
#     chunk: dict with fields:
#        - frames: list of Python strings
#        - label: int
#        - vid, clip: int
#        - side: 'left' or 'right'
#        - t_center, t_width: float
#     """

#     frame_paths = chunk["frames"]
#     label = chunk["label"]

#     # input(chunk)
#     # convert python list → tf constant
#     # path_tensor = tf.constant(frame_paths)
#     path_tensor = frame_paths

#     # load all 60 images
#     imgs = tf.map_fn(
#         lambda p: load_and_preprocess_image(p, img_size),
#         path_tensor,
#         fn_output_signature=tf.float32
#     )
#     # shape = (chunk_size, H, W, 3)

#     metadata = {
#         "vid": chunk["vid"],
#         "clip": chunk["clip"],
#         "side": chunk["side"],
#         "t_center": chunk["t_center"],
#         "t_width": chunk["t_width"]
#     }

#     # label = tf.constant(label, dtype=tf.int32)
#     return imgs, metadata, label

def parse_chunk(chunk, img_size=(432,768)):
    frame_paths = chunk["frames"]
    label = chunk["label"]

    imgs = tf.map_fn(
        lambda p: load_and_preprocess_image(p, img_size),
        frame_paths,
        fn_output_signature=tf.float32
    )

    metadata = {
        "frames": chunk['frames'],
        "vid": chunk["vid"],
        "clip": chunk["clip"],
        "side": chunk["side"],
        "status": chunk["status"],
        "status_id": chunk["status_id"],
        "t_center": chunk["t_center"],
        "t_width": chunk["t_width"],
        "start_idx": chunk["start_idx"],
        "end_idx": chunk["end_idx"],
    }

    return imgs, metadata, label

# chunk_samples.append({
#                 "frames": frame_paths,    # list of image paths (length chunk_size)
#                 "label": label,
#                 "side": side,
#                 "vid": vid,
#                 "clip": clip,
#                 "t_center": t_center,
#                 "t_width": t_width, #max(t_width, 0.4)
#             })
# {'frames': ['/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19754.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19755.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19756.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19757.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19758.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19759.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19760.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19761.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19762.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19763.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19764.jpg', 
#             '/home/vasantgc/venv/nba_proj/data/unseen_test_images/smarter_clips/clips_hmm_smooth_vid3_smart/vid3_clip_4_left/vid3_frame_19765.jpg'], 
#  'label': 1, 
#  'side': 'left', 
#  'vid': 3, 
#  'clip': 4, 
#  't_center': 0.11796536796536798, 
#  't_width': 0.023809523809523794}

# def build_tf_dataset_chunks(chunk_samples, batch_size, img_size=(432,768), num_workers=16, training = False):
#     # vids 8 and 10 are test vids
#     def gen():
#         for sample in chunk_samples:
#             yield sample

#     output_signature = {
#         "frames": tf.TensorSpec(shape=(None,), dtype=tf.string),  # list of frame paths
#         "label": tf.TensorSpec(shape=(), dtype=tf.int32),
#         "side":  tf.TensorSpec(shape=(), dtype=tf.string),
#         "vid":   tf.TensorSpec(shape=(), dtype=tf.int32),
#         "clip":  tf.TensorSpec(shape=(), dtype=tf.int32),
#         "t_center": tf.TensorSpec(shape=(), dtype=tf.float32),
#         "t_width":  tf.TensorSpec(shape=(), dtype=tf.float32),
#     }

#     ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

#     # shuffle dataset order
#     ds = ds.shuffle(len(chunk_samples), reshuffle_each_iteration=True) # 2048 was len(chunk_samples)

#     # map to parsed tensors (loads images, builds metadata dict)
#     ds = ds.map(lambda chunk: parse_chunk(chunk, img_size),
#                 num_parallel_calls=num_workers)

#     # batching & prefetch
#     ds = ds.batch(batch_size, drop_remainder=True)
#     ds = ds.prefetch(tf.data.AUTOTUNE)

#     return ds

#     # ds = tf.data.Dataset.from_tensor_slices(chunk_samples)
#     # ds = ds.shuffle(len(chunk_samples), reshuffle_each_iteration=True)

#     # # ds = ds.map(parse_chunk, num_parallel_calls=16)

#     # ds = ds.map(
#     #     lambda chunk: parse_chunk(chunk, img_size),
#     #     num_parallel_calls=num_workers
#     # )

#     # ds = ds.batch(batch_size, drop_remainder=True)
#     # ds = ds.prefetch(tf.data.AUTOTUNE)
#     # ds = tf.data.Dataset.from_generator(
#     #     lambda: chunk_samples,
#     #     output_types={
#     #         "frames": tf.string,
#     #         "label": tf.int32,
#     #         "side": tf.string,
#     #         "vid": tf.int32,
#     #         "clip": tf.int32,
#     #         "t_center": tf.float32,
#     #         "t_width": tf.float32,
#     #     }
#     # )

    

#     # ds = ds.shuffle(256)
#     # ds = ds.batch(batch_size, drop_remainder=True)
#     # ds = ds.prefetch(tf.data.AUTOTUNE)

#     return ds

def build_tf_dataset_chunks(chunk_samples, batch_size, img_size=(432,768), num_workers=16, training=False):
    def gen():
        for sample in chunk_samples:
            yield {
                "frames": sample["frames"],
                "label": sample["label"],
                "side": sample["side"],
                "status": sample["status"],
                "status_id": sample["status_id"],
                "vid": sample["vid"],
                "clip": sample["clip"],
                "t_center": sample["t_center"],
                "t_width": sample["t_width"],
                "start_idx": sample["start_idx"],
                "end_idx": sample["end_idx"],
            }

    output_signature = {
        "frames": tf.TensorSpec(shape=(None,), dtype=tf.string),
        "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        "side": tf.TensorSpec(shape=(), dtype=tf.string),
        "status": tf.TensorSpec(shape=(), dtype=tf.string),
        "status_id": tf.TensorSpec(shape=(), dtype=tf.int32),
        "vid": tf.TensorSpec(shape=(), dtype=tf.int32),
        "clip": tf.TensorSpec(shape=(), dtype=tf.int32),
        "t_center": tf.TensorSpec(shape=(), dtype=tf.float32),
        "t_width": tf.TensorSpec(shape=(), dtype=tf.float32),
        "start_idx": tf.TensorSpec(shape=(), dtype=tf.int32),
        "end_idx": tf.TensorSpec(shape=(), dtype=tf.int32),
    }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    ds = ds.shuffle(len(chunk_samples), seed=1234, reshuffle_each_iteration=False)

    ds = ds.map(
        lambda chunk: parse_chunk(chunk, img_size),
        num_parallel_calls=num_workers
    )

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# def chunk_generator(chunk_samples):
#     for c in chunk_samples:
#         yield {
#             "frames": c["frames"],         # python list of strings
#             "label": c["label"],
#             "side": c["side"],
#             "vid": c["vid"],
#             "clip": c["clip"],
#             "t_center": c["t_center"],
#             "t_width": c["t_width"],
#         }
        