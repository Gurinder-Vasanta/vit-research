import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from loader import preprocess_frame


def batch_generator(samples, batch_size=32, num_workers=32):
    """
    Samples: list of dicts: {pth, vid_num, clip_num, side, t_norm, label}
    """
    N = len(samples)
    idx = 0

    pool = Pool(num_workers)

    while idx < N:
        batch_samples = samples[idx : idx + batch_size]

        paths = [s["pth"] for s in batch_samples]
        frames = pool.map(preprocess_frame, paths)
        frames = np.array(frames, dtype=np.float32)

        metadata = {
            "vid":    np.array([s["vid_num"]  for s in batch_samples], dtype=np.int32),
            "clip":   np.array([s["clip_num"] for s in batch_samples], dtype=np.int32),
            "side":   np.array([s["side"]     for s in batch_samples], dtype=object),
            "t_norm": np.array([s["t_norm"]   for s in batch_samples], dtype=np.float32)
        }

        labels = np.array([s["label"] for s in batch_samples], dtype=np.int32)

        yield frames, metadata, labels

        idx += batch_size

    pool.close()
    pool.join()



def build_tf_dataset(samples, batch_size=32, num_workers=32):
    """
    Wrap the Python generator into a tf.data.Dataset.
    """

    output_signature = (
        tf.TensorSpec(shape=(None, 432, 768, 3), dtype=tf.float32),  # frames batch
        {
            "vid": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "clip": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "side": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "t_norm": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    return (
        tf.data.Dataset.from_generator(
            lambda: batch_generator(samples, batch_size, num_workers),
            output_signature=output_signature
        )
        .prefetch(tf.data.AUTOTUNE)
    )