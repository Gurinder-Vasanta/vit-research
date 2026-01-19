import numpy as np

class RattChunkRetriever:
    def __init__(self, collection, top_k=10, search_k=200):
        self.collection = collection
        self.top_k = top_k
        self.search_k = search_k

    def __call__(self, chunk_embs, metadata):
        if hasattr(chunk_embs, "numpy"):
            q_np = chunk_embs.numpy()
        else:
            q_np = chunk_embs

        B = q_np.shape[0]
        retrieved_all = []

        for i in range(B):
            vid      = int(metadata["vid"][i].numpy())
            side     = metadata["side"][i].numpy().decode()
            t_center = float(metadata["t_center"][i].numpy())
            t_width  = float(metadata["t_width"][i].numpy())

            q_start = t_center - t_width / 2
            q_end   = t_center + t_width / 2

            result = self.collection.query(
                query_embeddings=[q_np[i].tolist()],
                n_results=self.search_k,
                where={
                    "$and": [
                        # {"vid_num": {"$ne": vid}},
                        {"side": side},
                        {"t_center": {"$gte": q_start}},
                        {"t_center": {"$lte": q_end}},
                    ]
                },
                include=["embeddings"]
            )

            retrieved_vecs = result["embeddings"][0]

            if len(retrieved_vecs) >= self.top_k:
                retrieved_vecs = retrieved_vecs[:self.top_k]
            else:
                D = q_np.shape[1]
                pad = np.zeros((self.top_k - len(retrieved_vecs), D), dtype=np.float32)
                retrieved_vecs = np.vstack([retrieved_vecs, pad])

            retrieved_vecs = np.array(retrieved_vecs, dtype=np.float32)
            retrieved_vecs /= (np.linalg.norm(retrieved_vecs, axis=1, keepdims=True) + 1e-8)

            retrieved_all.append(retrieved_vecs)

        return np.stack(retrieved_all, axis=0)

