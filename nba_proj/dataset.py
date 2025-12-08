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

def load_samples(train_vids, stride = 1):
    already_labelled = pd.read_csv("clips_label.csv")
    
    samples = []

    for vid in train_vids:
        # clip_root = f"/home/.../clips_finalized_{vid}"

        clip_root = f'/home/vasantgc/venv/nba_proj/data/unseen_test_images/clips_finalized_{vid}'
        clips = sorted(os.listdir(clip_root))

        for clip in clips:
            clip_path = os.path.join(clip_root, clip)
            frames = sorted(os.listdir(clip_path))

            # find label
            label_row = already_labelled[
                already_labelled["clip_path"] == clip_path
            ]
            # print(len(np.array(label_row)[0]))
            if label_row.empty or pd.isna(label_row['label'].iloc[0]):
                continue
            clip_label = int(label_row["label"])

            num_frames = len(frames)
            stride_counter = 0
            for i, f in enumerate(frames, start=1):
                fpath = os.path.join(clip_path, f)
                
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
                        "label": clip_label
                    })

                    stride_counter = 0
    # input(np.array(samples))
    return samples

def build_chunks(frame_samples, chunk_size=60):
    """
    frame_samples: list of dictionaries (one per frame)
    chunk_size: number of frames to group into one chunk

    Returns: list of chunk dictionaries
    """

    # --- Group frames by clip ---
    clips = {}
    for s in frame_samples:
        key = (s["vid_num"], s["clip_num"])
        if key not in clips:
            clips[key] = []
        clips[key].append(s)

    # --- Sort frames within each clip ---
    for key in clips:
        clips[key].sort(key=lambda x: x["t_norm"])

    # --- Build chunks ---
    chunk_samples = []

    for (vid, clip), frames in clips.items():
        total_frames = len(frames)
        label = frames[0]["label"]     # all frames share the clip label
        side = frames[0]["side"]       # same within a clip

        # slide with step = chunk_size (no overlap for now)
        for i in range(0, total_frames, chunk_size):
            sub = frames[i : i + chunk_size]
            if len(sub) < chunk_size:
                # OPTIONAL: skip incomplete chunk
                continue

            # frame paths
            frame_paths = [f["pth"] for f in sub]

            # compute temporal window
            t_vals = [f["t_norm"] for f in sub]
            t_center = float(sum(t_vals) / len(t_vals))
            t_width = float(max(t_vals) - min(t_vals))  # ~60 frames worth

            chunk_samples.append({
                "frames": frame_paths,    # list of image paths (length chunk_size)
                "label": label,
                "side": side,
                "vid": vid,
                "clip": clip,
                "t_center": t_center,
                "t_width": t_width, #max(t_width, 0.4)
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

def parse_chunk(chunk, img_size=(432,768)):
    """
    chunk: dict with fields:
       - frames: list of Python strings
       - label: int
       - vid, clip: int
       - side: 'left' or 'right'
       - t_center, t_width: float
    """

    frame_paths = chunk["frames"]
    label = chunk["label"]

    # input(chunk)
    # convert python list → tf constant
    # path_tensor = tf.constant(frame_paths)
    path_tensor = frame_paths

    # load all 60 images
    imgs = tf.map_fn(
        lambda p: load_and_preprocess_image(p, img_size),
        path_tensor,
        fn_output_signature=tf.float32
    )
    # shape = (chunk_size, H, W, 3)

    metadata = {
        "vid": chunk["vid"],
        "clip": chunk["clip"],
        "side": chunk["side"],
        "t_center": chunk["t_center"],
        "t_width": chunk["t_width"]
    }

    # label = tf.constant(label, dtype=tf.int32)
    return imgs, metadata, label

def build_tf_dataset_chunks(chunk_samples, batch_size, img_size=(432,768), num_workers=16):
    def gen():
        for sample in chunk_samples:
            yield sample

    output_signature = {
        "frames": tf.TensorSpec(shape=(None,), dtype=tf.string),  # list of frame paths
        "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        "side":  tf.TensorSpec(shape=(), dtype=tf.string),
        "vid":   tf.TensorSpec(shape=(), dtype=tf.int32),
        "clip":  tf.TensorSpec(shape=(), dtype=tf.int32),
        "t_center": tf.TensorSpec(shape=(), dtype=tf.float32),
        "t_width":  tf.TensorSpec(shape=(), dtype=tf.float32),
    }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # shuffle dataset order
    ds = ds.shuffle(len(chunk_samples), reshuffle_each_iteration=True)

    # map to parsed tensors (loads images, builds metadata dict)
    ds = ds.map(lambda chunk: parse_chunk(chunk, img_size),
                num_parallel_calls=num_workers)

    # batching & prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

    # ds = tf.data.Dataset.from_tensor_slices(chunk_samples)
    # ds = ds.shuffle(len(chunk_samples), reshuffle_each_iteration=True)

    # # ds = ds.map(parse_chunk, num_parallel_calls=16)

    # ds = ds.map(
    #     lambda chunk: parse_chunk(chunk, img_size),
    #     num_parallel_calls=num_workers
    # )

    # ds = ds.batch(batch_size, drop_remainder=True)
    # ds = ds.prefetch(tf.data.AUTOTUNE)
    # ds = tf.data.Dataset.from_generator(
    #     lambda: chunk_samples,
    #     output_types={
    #         "frames": tf.string,
    #         "label": tf.int32,
    #         "side": tf.string,
    #         "vid": tf.int32,
    #         "clip": tf.int32,
    #         "t_center": tf.float32,
    #         "t_width": tf.float32,
    #     }
    # )

    

    # ds = ds.shuffle(256)
    # ds = ds.batch(batch_size, drop_remainder=True)
    # ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def chunk_generator(chunk_samples):
    for c in chunk_samples:
        yield {
            "frames": c["frames"],         # python list of strings
            "label": c["label"],
            "side": c["side"],
            "vid": c["vid"],
            "clip": c["clip"],
            "t_center": c["t_center"],
            "t_width": c["t_width"],
        }

# def batch_generator(samples, batch_size=32, num_workers=32):
#     """
#     Samples: list of dicts: {pth, vid_num, clip_num, side, t_norm, label}
#     """
#     N = len(samples)
#     idx = 0

#     pool = Pool(num_workers)

#     while idx < N:
#         batch_samples = samples[idx : idx + batch_size]

#         paths = [s["pth"] for s in batch_samples]
#         frames = pool.map(preprocess_frame, paths)
#         frames = np.array(frames, dtype=np.float32)

#         metadata = {
#             "vid":    np.array([s["vid_num"]  for s in batch_samples], dtype=np.int32),
#             "clip":   np.array([s["clip_num"] for s in batch_samples], dtype=np.int32),
#             "side":   np.array([s["side"]     for s in batch_samples], dtype=object),
#             "t_norm": np.array([s["t_norm"]   for s in batch_samples], dtype=np.float32)
#         }

#         labels = np.array([s["label"] for s in batch_samples], dtype=np.int32)

#         yield frames, metadata, labels

#         idx += batch_size

#     pool.close()
#     pool.join()



# def build_tf_dataset(samples, batch_size=32, num_workers=32):
#     """
#     Wrap the Python generator into a tf.data.Dataset.
#     """

#     output_signature = (
#         tf.TensorSpec(shape=(None, 432, 768, 3), dtype=tf.float32),  # frames batch
#         {
#             "vid": tf.TensorSpec(shape=(None,), dtype=tf.int32),
#             "clip": tf.TensorSpec(shape=(None,), dtype=tf.int32),
#             "side": tf.TensorSpec(shape=(None,), dtype=tf.string),
#             "t_norm": tf.TensorSpec(shape=(None,), dtype=tf.float32),
#         },
#         tf.TensorSpec(shape=(None,), dtype=tf.int32),
#     )

#     return (
#         tf.data.Dataset.from_generator(
#             lambda: batch_generator(samples, batch_size, num_workers),
#             output_signature=output_signature
#         )
#         .prefetch(tf.data.AUTOTUNE)
#     )