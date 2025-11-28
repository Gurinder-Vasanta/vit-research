import numpy as np

class FrameRetriever:
    def __init__(self, chroma_collection, top_k=50, search_k=100):
        self.collection = chroma_collection
        self.top_k = top_k
        self.search_k = search_k

    def __call__(self, cls_embeddings_np, metadata_batch):
        """
        cls_embeddings_np: (B, D) numpy array
        metadata_batch: list of length B, each:
            {
              "vid": int,
              "clip": int,
              "side": str,
              "t_norm": float
            }
        Returns: np.ndarray of shape (B, top_k, D)
        """
        batch_retrieved = []
        
        batch_vids = metadata_batch['vid'].numpy()
        batch_clips = metadata_batch['clip'].numpy()
        batch_sides = metadata_batch['side'].numpy()
        batch_tnorms = metadata_batch['t_norm'].numpy()

        for emb_vec, vid, clip, side, tnorm in zip(cls_embeddings_np, batch_vids, batch_clips, batch_sides, batch_tnorms):
            # input(emb_vec)
            # emb = emb_vec.reshape(1, -1).tolist()
            emb = emb_vec.numpy()[None, :]

            side = side.decode('utf-8')
            # input([emb_vec,vid,clip,side,tnorm])
            # input(metadata_batch.values())
            # input(metadata_batch['vid'].numpy())
# {'vid': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([2, 2, 2, 2], dtype=int32)>, 
# 'clip': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([100, 100, 100, 100], dtype=int32)>, 
# 'side': <tf.Tensor: shape=(4,), dtype=string, numpy=array([b'left', b'left', b'left', b'left'], dtype=object)>, 
# 't_norm': <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.0020284 , 0.0040568 , 0.00608519, 0.00811359], dtype=float32)>}
            # input(cls_embeddings_np)
            # input(metadata_batch['vid'])
            # input(meta)
            # input(metadata_batch)

            # vid = int(meta["vid"])
            # clip = int(meta["clip"])
            # side = meta["side"]
            # tnorm = float(meta["t_norm"])

            results = self.collection.query(
                query_embeddings=emb,
                n_results=self.search_k,
                where={
                    "$and": [
                        {"side": side},
                        {"t_norm": {"$gte": tnorm - 0.05}},
                        {"t_norm": {"$lte": tnorm + 0.05}},
                        {"vid": {"$ne": int(vid)}}
                    ]
                },
                include=["embeddings", "metadatas", "distances"]
            )

            retrieved = []

            for emb_, m in zip(results["embeddings"][0], results["metadatas"][0]):
                # print(len(emb_))
                # print(len(m))
                # input('stop')
                # print(emb_,m)
                # input('stop')
                # input(emb_)
                # need to add the tnorm condition in here as well (no you dont, its in the query)
                if (m["clip_num"] != clip):
                    retrieved.append(emb_)
                if len(retrieved) == self.top_k:
                    break

            retrieved = np.asarray(retrieved, dtype=np.float32)
            batch_retrieved.append(retrieved)
        # print(batch_retrieved)
        # input('batch_retrieved')
        return np.stack(batch_retrieved, axis=0)  # (B, top_k, D)