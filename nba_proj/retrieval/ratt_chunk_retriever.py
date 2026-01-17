import numpy as np

class RattChunkRetriever:
    def __init__(self, collection, top_k=10, search_k=200):
        self.collection = collection
        self.top_k = top_k
        self.search_k = search_k

    def __call__(self, chunk_embs, metadata):
        """
        chunk_embs: (B, 768) normalized projected embeddings
        metadata: dict of tf tensors:
            metadata["vid"]      shape (B,)
            metadata["side"]     shape (B,)
            metadata["t_center"] shape (B,)
            metadata["t_width"]  shape (B,)
        """

        # Convert to numpy
        if hasattr(chunk_embs, "numpy"):
            q_np = chunk_embs.numpy()
        else:
            q_np = chunk_embs
        
        B = q_np.shape[0]
        retrieved_all = []

        for i in range(B):
            # ---- Extract metadata for this chunk ----
            vid      = int(metadata["vid"][i].numpy())
            side     = metadata["side"][i].numpy().decode()
            t_center = float(metadata["t_center"][i].numpy())
            t_width  = float(metadata["t_width"][i].numpy())

            # local temporal window
            t_min = t_center - (t_width / 2)
            t_max = t_center + (t_width / 2)

            # print(metadata)
            # ---- Query the Chroma DB with metadata filters ----
            result = self.collection.query(
                query_embeddings=[q_np[i].tolist()],
                n_results=self.search_k,
                where={
                    "$and": [
                        {"vid_num": {"$ne": vid}},      # exclude same video
                        {"side": side},                 # same direction
                        {"t_norm": {"$gte": t_min}},    # local window
                        {"t_norm": {"$lte": t_max}}
                    ]
                },
                include=["embeddings"]
            )

            retrieved_vecs = result["embeddings"][0]

            # print(len(retrieved_vecs))
            # ---- Keep only top_k ----
            if len(retrieved_vecs) >= self.top_k:
                retrieved_vecs = retrieved_vecs[:self.top_k]
            else:
                # zero pad if needed
                D = q_np.shape[1]
                pad = np.zeros((self.top_k - len(retrieved_vecs), D), dtype=np.float32)
                retrieved_vecs = np.vstack([retrieved_vecs, pad])

            # ---- Normalize for safety ----
            retrieved_vecs = np.array(retrieved_vecs, dtype=np.float32)
            retrieved_vecs /= (np.linalg.norm(retrieved_vecs, axis=1, keepdims=True) + 1e-8)

            retrieved_all.append(retrieved_vecs)

        return np.stack(retrieved_all, axis=0)  # (B, top_k, 768)
