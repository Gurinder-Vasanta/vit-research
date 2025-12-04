import numpy as np

class FrameRetriever:
    def __init__(self, collection, top_k=50, search_k=200):
        self.collection = collection
        self.top_k = top_k
        self.search_k = search_k

    def __call__(self, chunk_embs, metadata):
        """
        chunk_embs: (B, 768)   raw ViT chunk embeddings from training
        metadata: dict with:
            - vid          (B,)   int
            - side         (B,)   tf.string
            - t_center     (B,)   float
            - t_width      (B,)   float
        """

        B = chunk_embs.shape[0]
        q_np = chunk_embs.numpy()
        out = []

        for i in range(B):

            vid      = int(metadata["vid"][i].numpy())
            side     = metadata["side"][i].numpy().decode()
            t_center = float(metadata["t_center"][i].numpy())
            t_width  = float(metadata["t_width"][i].numpy())

            t_min = t_center - (t_width / 2)
            t_max = t_center + (t_width / 2)

            # Query the NEW enriched DB
            results = self.collection.query(
                query_embeddings=q_np[i:i+1],
                n_results=self.search_k,
                where={
                    "$and": [
                        {"vid_num": {"$ne": vid}},
                        {"side": side},
                        {"t_norm": {"$gte": t_min}},
                        {"t_norm": {"$lte": t_max}}
                    ]
                },
                include=["embeddings"]
            )

            embs = results["embeddings"][0][:self.top_k]

            # pad if too few results
            if len(embs) < self.top_k:
                D = q_np.shape[1]
                pad = np.zeros((self.top_k - len(embs), D), dtype=np.float32)
                embs = np.vstack([embs, pad])

            out.append(np.asarray(embs, dtype=np.float32))

        return np.stack(out, axis=0)